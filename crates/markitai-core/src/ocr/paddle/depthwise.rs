//! Depthwise convolutions (one filter per channel) spread over the OCR
//! threads. tract runs its own depthwise convolution on one thread, and in
//! a recognizer, whose line width varies, as a general convolution: the
//! PP-OCR detector spent over half its time there. Here each channel is
//! computed on its own, in tract's order (the bias, then each tap from the
//! top left), so the result does not depend on the number of threads.

use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::*;
use tract_onnx::tract_core::ops::cnn::{Conv, KernelFormat};
use tract_onnx::tract_core::ops::nn::DataFormat;

/// A depthwise convolution of `f32` channels laid out as (N)CHW, with a
/// constant kernel (OIHW) and bias: the same values as `conv` computes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct Depthwise {
    conv: Conv,
}

/// Replace the depthwise convolutions of a decluttered `model` by
/// [`Depthwise`]; other nodes are left as they are. Returns how many were
/// replaced.
pub(super) fn split(model: &mut TypedModel) -> TractResult<usize> {
    let mut replaced = 0;
    for id in 0..model.nodes.len() {
        if let Some(op) = Depthwise::of(model, &model.nodes[id])? {
            model.nodes[id].op = Box::new(op);
            replaced += 1;
        }
    }
    Ok(replaced)
}

impl Depthwise {
    fn of(model: &TypedModel, node: &TypedNode) -> TractResult<Option<Depthwise>> {
        let Some(conv) = node.op_as::<Conv>() else {
            return Ok(None);
        };
        if node.inputs.len() != 3 {
            return Ok(None);
        }
        let spec = &conv.pool_spec;
        let input = model.outlet_fact(node.inputs[0])?;
        let constant = |index: usize| -> TractResult<bool> {
            let fact = model.outlet_fact(node.inputs[index])?;
            Ok(fact.konst.is_some() && fact.datum_type == f32::datum_type())
        };
        let depthwise = conv.q_params.is_none()
            && input.datum_type == f32::datum_type()
            && matches!(
                (spec.data_format, input.rank()),
                (DataFormat::NCHW, 4) | (DataFormat::CHW, 3)
            )
            && conv.kernel_fmt == KernelFormat::OIHW
            && spec.kernel_shape.len() == 2
            && conv.group > 1
            && conv.group == spec.input_channels
            && conv.group == spec.output_channels
            && constant(1)?
            && constant(2)?;
        Ok(depthwise.then(|| Depthwise { conv: conv.clone() }))
    }

    fn geometry(&self, height: usize, width: usize) -> Geometry {
        let spec = &self.conv.pool_spec;
        let (kh, kw) = (spec.kernel_shape[0], spec.kernel_shape[1]);
        let strides = spec.strides.clone().unwrap_or_else(|| tvec!(1, 1));
        let dilations = spec.dilations.clone().unwrap_or_else(|| tvec!(1, 1));
        let padded = spec
            .padding
            .compute(&[height, width], &[kh, kw], &dilations, &strides);
        Geometry {
            height,
            width,
            out_height: padded[0].convoluted,
            out_width: padded[1].convoluted,
            kernel_height: kh,
            kernel_width: kw,
            stride_y: strides[0],
            stride_x: strides[1],
            dilation_y: dilations[0],
            dilation_x: dilations[1],
            top: padded[0].pad_before,
            left: padded[1].pad_before,
        }
    }
}

impl Op for Depthwise {
    fn name(&self) -> StaticName {
        "MarkitaiDepthwise".into()
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalOp for Depthwise {
    op_out_of_plan!();

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, kernel, bias) = args_3!(inputs);
        let (batch, channels, height, width) = match *input.shape() {
            [n, c, h, w] => (n, c, h, w),
            [c, h, w] => (1, c, h, w),
            _ => bail!("a depthwise convolution input is not (N)CHW"),
        };
        let geometry = self.geometry(height, width);
        let taps = geometry.kernel_height * geometry.kernel_width;
        let (input, kernel, bias) = (
            input.try_as_plain_ram()?,
            kernel.try_as_plain_ram()?,
            bias.try_as_plain_ram()?,
        );
        let (input, kernel, bias) = (
            input.as_slice::<f32>()?,
            kernel.as_slice::<f32>()?,
            bias.as_slice::<f32>()?,
        );
        ensure!(
            input.len() == batch * channels * height * width
                && kernel.len() == channels * taps
                && (bias.len() == channels || bias.len() == 1),
            "a depthwise convolution's inputs do not match its shape"
        );
        let mut shape: TVec<usize> = tvec!(channels, geometry.out_height, geometry.out_width);
        if self.conv.pool_spec.data_format == DataFormat::NCHW {
            shape.insert(0, batch);
        }
        let mut output = Tensor::zero::<f32>(&shape)?;
        let plane = geometry.out_height * geometry.out_width;
        if plane > 0 {
            let mut view = output.try_as_plain_ram_mut()?;
            let values = view.as_slice_mut::<f32>()?;
            let work = values.len() * taps;
            tract_linalg::multithread::par_chunks_mut(values, plane, work, |first, chunk| {
                for (offset, out) in chunk.chunks_exact_mut(plane).enumerate() {
                    let index = first + offset;
                    let channel = index % channels;
                    geometry.channel(
                        &input[index * height * width..][..height * width],
                        &kernel[channel * taps..][..taps],
                        bias[if bias.len() == 1 { 0 } else { channel }],
                        out,
                    );
                }
                Ok(())
            })?;
        }
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for Depthwise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        self.conv.output_facts(inputs)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        self.conv.cost(inputs)
    }

    as_op!();
}

/// One channel's convolution, in input and output pixels.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride_y: usize,
    stride_x: usize,
    dilation_y: usize,
    dilation_x: usize,
    /// Padding rows above and columns left of the input.
    top: usize,
    left: usize,
}

impl Geometry {
    /// The output columns that read input column `x * stride_x + offset`
    /// inside the row: the others read padding.
    fn columns(&self, offset: isize) -> std::ops::Range<usize> {
        let stride = self.stride_x as isize;
        let width = self.width as isize;
        let first = if offset >= 0 {
            0
        } else {
            (-offset + stride - 1) / stride
        };
        let end = if width - 1 - offset < 0 {
            0
        } else {
            ((width - 1 - offset) / stride + 1).min(self.out_width as isize)
        };
        first as usize..end.max(first) as usize
    }

    /// Convolve one channel's `input` plane with its `taps` into `output`.
    fn channel(&self, input: &[f32], taps: &[f32], bias: f32, output: &mut [f32]) {
        for (y, row) in output.chunks_exact_mut(self.out_width).enumerate() {
            row.fill(bias);
            for ky in 0..self.kernel_height {
                let source =
                    (y * self.stride_y + ky * self.dilation_y) as isize - self.top as isize;
                if source < 0 || source >= self.height as isize {
                    continue;
                }
                let line = &input[source as usize * self.width..][..self.width];
                for kx in 0..self.kernel_width {
                    let tap = taps[ky * self.kernel_width + kx];
                    let offset = (kx * self.dilation_x) as isize - self.left as isize;
                    let columns = self.columns(offset);
                    if columns.is_empty() {
                        continue;
                    }
                    if self.stride_x == 1 {
                        let start = (columns.start as isize + offset) as usize;
                        let read = &line[start..start + columns.len()];
                        for (out, value) in row[columns].iter_mut().zip(read) {
                            *out += tap * *value;
                        }
                    } else {
                        for x in columns {
                            let column = (x as isize * self.stride_x as isize + offset) as usize;
                            row[x] += tap * line[column];
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_onnx::tract_core::ops::cnn::{PaddingSpec, PoolSpec};

    /// A model of one depthwise convolution over a `shape` input.
    fn model(
        shape: &[usize],
        kernel: [usize; 2],
        padding: PaddingSpec,
        strides: [usize; 2],
        dilations: [usize; 2],
        bias_per_channel: bool,
    ) -> TypedModel {
        let channels = shape[shape.len() - 3];
        let format = if shape.len() == 4 {
            DataFormat::NCHW
        } else {
            DataFormat::CHW
        };
        let mut model = TypedModel::default();
        let input = model.add_source("input", f32::fact(shape)).expect("source");
        let taps = kernel[0] * kernel[1];
        let weights: Vec<f32> = (0..channels * taps)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
            .collect();
        let weights =
            Tensor::from_shape(&[channels, 1, kernel[0], kernel[1]], &weights).expect("kernel");
        let bias = if bias_per_channel {
            let values: Vec<f32> = (0..channels).map(|c| c as f32 * 0.25 - 1.0).collect();
            Tensor::from_shape(&[channels], &values).expect("bias")
        } else {
            tensor0(0.5f32)
        };
        let weights = model.add_const("kernel", weights).expect("kernel");
        let bias = model.add_const("bias", bias).expect("bias");
        let conv = Conv::new(
            PoolSpec::new(
                format,
                tvec!(kernel[0], kernel[1]),
                padding,
                Some(tvec!(dilations[0], dilations[1])),
                Some(tvec!(strides[0], strides[1])),
                channels,
                channels,
            ),
            KernelFormat::OIHW,
            channels,
            None,
        );
        let output = model
            .wire_node("conv", conv, &[input, weights, bias])
            .expect("conv");
        model.select_output_outlets(&output).expect("output");
        model
    }

    fn run(model: TypedModel, input: &Tensor, threads: usize) -> Vec<f32> {
        let plan = model
            .into_optimized()
            .and_then(|model| model.into_runnable())
            .expect("plan");
        let executor = if threads > 1 {
            tract_linalg::multithread::Executor::multithread(threads)
        } else {
            tract_linalg::multithread::Executor::SingleThread
        };
        let outputs = tract_linalg::multithread::multithread_tract_scope(executor, || {
            plan.run(tvec!(input.clone().into()))
        })
        .expect("run");
        let view = outputs[0].try_as_plain_ram().expect("plain");
        view.as_slice::<f32>().expect("f32").to_vec()
    }

    /// One convolution geometry the replacement must reproduce.
    struct Case {
        shape: &'static [usize],
        kernel: [usize; 2],
        padding: PaddingSpec,
        strides: [usize; 2],
        dilations: [usize; 2],
        bias: bool,
    }

    fn case(shape: &'static [usize], kernel: usize, padding: PaddingSpec) -> Case {
        Case {
            shape,
            kernel: [kernel, kernel],
            padding,
            strides: [1, 1],
            dilations: [1, 1],
            bias: true,
        }
    }

    #[test]
    fn depthwise_convolutions_match_tract_on_any_number_of_threads() {
        let explicit = |pad: usize| PaddingSpec::Explicit(tvec!(pad, pad), tvec!(pad, pad));
        let cases = [
            case(&[1, 8, 40, 50], 3, PaddingSpec::SameUpper),
            Case {
                bias: false,
                ..case(&[6, 17, 23], 3, PaddingSpec::SameUpper)
            },
            case(&[1, 5, 33, 40], 7, PaddingSpec::SameUpper),
            Case {
                strides: [2, 2],
                ..case(&[1, 4, 19, 31], 3, explicit(1))
            },
            Case {
                strides: [2, 1],
                bias: false,
                ..case(&[1, 4, 20, 30], 5, explicit(2))
            },
            Case {
                dilations: [2, 2],
                ..case(&[1, 3, 21, 26], 3, explicit(2))
            },
            Case {
                kernel: [3, 5],
                strides: [1, 2],
                ..case(&[2, 3, 9, 11], 3, PaddingSpec::Valid)
            },
            case(&[1, 2, 2, 3], 5, explicit(2)),
        ];
        for Case {
            shape,
            kernel,
            padding,
            strides,
            dilations,
            bias,
        } in &cases
        {
            let len: usize = shape.iter().product();
            let values: Vec<f32> = (0..len)
                .map(|i| ((i * 7919 % 251) as f32 / 125.0 - 1.0) * 0.9)
                .collect();
            let input = Tensor::from_shape(shape, &values).expect("input");
            let reference = model(shape, *kernel, padding.clone(), *strides, *dilations, *bias);
            let expected = run(reference.clone(), &input, 1);
            let mut ours = reference.into_decluttered().expect("declutter");
            assert_eq!(split(&mut ours).expect("split"), 1, "{shape:?} {kernel:?}");
            for threads in [1, 4] {
                let actual = run(ours.clone(), &input, threads);
                assert_eq!(actual.len(), expected.len(), "{shape:?} {kernel:?}");
                for (a, e) in actual.iter().zip(&expected) {
                    assert!(
                        (a - e).abs() <= 1e-5 * e.abs().max(1.0),
                        "{shape:?} {kernel:?} {strides:?} {dilations:?} on {threads} threads: {a} != {e}"
                    );
                }
            }
        }
    }

    #[test]
    fn only_float_depthwise_convolutions_are_replaced() {
        let mut ordinary = TypedModel::default();
        let input = ordinary
            .add_source("input", f32::fact([1, 4, 8, 8]))
            .expect("source");
        let weights = ordinary
            .add_const(
                "kernel",
                Tensor::zero::<f32>(&[4, 2, 3, 3]).expect("kernel"),
            )
            .expect("kernel");
        let bias = ordinary.add_const("bias", tensor0(0f32)).expect("bias");
        let grouped = Conv::new(
            PoolSpec::new(
                DataFormat::NCHW,
                tvec!(3, 3),
                PaddingSpec::SameUpper,
                None,
                None,
                4,
                4,
            ),
            KernelFormat::OIHW,
            2,
            None,
        );
        let output = ordinary
            .wire_node("grouped", grouped, &[input, weights, bias])
            .expect("conv");
        ordinary.select_output_outlets(&output).expect("output");
        assert_eq!(split(&mut ordinary).expect("split"), 0);
    }
}
