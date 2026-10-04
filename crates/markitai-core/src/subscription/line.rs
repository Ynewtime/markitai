//! Bounded newline-delimited output shared by the ChatGPT and Claude runtimes.

use super::FailureKind;
use std::io::BufRead;

const LINE_LIMIT: usize = 16 * 1024 * 1024;
const STREAM_LIMIT: usize = 64 * 1024 * 1024;

pub(super) fn read_line(
    reader: &mut impl BufRead,
    total: &mut usize,
) -> Result<Option<Vec<u8>>, FailureKind> {
    let mut line = Vec::new();
    loop {
        let bytes = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(FailureKind::Transport),
        };
        if bytes.is_empty() {
            return if line.is_empty() {
                Ok(None)
            } else {
                Ok(Some(line))
            };
        }
        let count = bytes
            .iter()
            .position(|b| *b == b'\n')
            .map_or(bytes.len(), |i| i + 1);
        if line.len().saturating_add(count) > LINE_LIMIT
            || total.saturating_add(count) > STREAM_LIMIT
        {
            return Err(FailureKind::ResourceLimit);
        }
        let ended = bytes[count - 1] == b'\n';
        line.extend_from_slice(&bytes[..count]);
        *total += count;
        reader.consume(count);
        if ended {
            return Ok(Some(line));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Read};

    struct InterruptedChunks {
        bytes: std::io::Cursor<Vec<u8>>,
        interrupt: bool,
    }
    impl Read for InterruptedChunks {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            self.interrupt = !self.interrupt;
            if self.interrupt {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let size = output.len().min(2);
            self.bytes.read(&mut output[..size])
        }
    }

    #[test]
    fn interrupted_pipe_reads_preserve_frames_counts_and_stream_limits() {
        let bytes = b"{\"x\":1}\n{}".to_vec();
        let mut reader = BufReader::new(InterruptedChunks {
            bytes: std::io::Cursor::new(bytes.clone()),
            interrupt: false,
        });
        let mut total = 0;
        assert_eq!(
            read_line(&mut reader, &mut total).unwrap().unwrap(),
            b"{\"x\":1}\n"
        );
        assert_eq!(read_line(&mut reader, &mut total).unwrap().unwrap(), b"{}");
        assert!(read_line(&mut reader, &mut total).unwrap().is_none());
        assert_eq!(total, bytes.len());

        let mut reader = BufReader::new(InterruptedChunks {
            bytes: std::io::Cursor::new(b"{}\n".to_vec()),
            interrupt: false,
        });
        let mut total = STREAM_LIMIT - 2;
        assert_eq!(
            read_line(&mut reader, &mut total),
            Err(FailureKind::ResourceLimit)
        );
    }
}
