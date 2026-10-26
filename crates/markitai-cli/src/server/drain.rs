//! A refusal that reaches the client through a proxy. A server that answers
//! before it has read an upload (a foreign origin, a missing token, too many
//! items) and closes leaves unread bytes behind: its system resets the
//! connection, and a proxy still sending the body (nginx, a Cloudflare tunnel)
//! reports 502 instead of the answer. So a refused request's unread body is
//! read and dropped first, up to the upload size limit and for a bounded time,
//! and the answer closes the connection. A larger body, declared or sent, is
//! not read: such a refusal stays as cheap as before.
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    body::{Body, BodyDataStream, Bytes},
    extract::Request,
    http::{HeaderValue, header},
    middleware::Next,
    response::Response,
};
use futures_util::{Stream, StreamExt};

use super::types::MAX_UPLOAD;

/// The most a refused request's body is read: one upload's limit.
const DRAIN_LIMIT: u64 = MAX_UPLOAD as u64;
/// How long a refused request's body may take to arrive.
const DRAIN_TIME: Duration = Duration::from_secs(120);

struct Shared {
    stream: BodyDataStream,
    read: u64,
    ended: bool,
}

/// The request body the routes see; the original stays reachable from here.
struct Tracked(Arc<Mutex<Shared>>);

impl Stream for Tracked {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut shared = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let next = shared.stream.poll_next_unpin(context);
        match &next {
            Poll::Ready(Some(Ok(bytes))) => shared.read += bytes.len() as u64,
            Poll::Ready(None) => shared.ended = true,
            _ => {}
        }
        next
    }
}

pub(super) async fn read_refused_bodies(request: Request, next: Next) -> Response {
    let headers = request.headers();
    let declared = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let streamed = headers.contains_key(header::TRANSFER_ENCODING);
    // Nothing to read, or more than a refusal may read.
    let readable = match declared {
        Some(size) => size > 0 && size <= DRAIN_LIMIT,
        None => streamed,
    };
    if !readable {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let shared = Arc::new(Mutex::new(Shared {
        stream: body.into_data_stream(),
        read: 0,
        ended: false,
    }));
    let request = Request::from_parts(parts, Body::from_stream(Tracked(shared.clone())));
    let mut response = next.run(request).await;
    if response.status().is_success() || response.status().is_redirection() {
        return response;
    }
    let (mut stream, read, ended) = {
        let mut shared = shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let stream = std::mem::replace(&mut shared.stream, Body::empty().into_data_stream());
        (stream, shared.read, shared.ended)
    };
    if ended {
        return response;
    }
    let _ = tokio::time::timeout(DRAIN_TIME, async {
        let mut read = read;
        while let Some(Ok(bytes)) = stream.next().await {
            read += bytes.len() as u64;
            if read > DRAIN_LIMIT {
                break;
            }
        }
    })
    .await;
    response
        .headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("close"));
    response
}
