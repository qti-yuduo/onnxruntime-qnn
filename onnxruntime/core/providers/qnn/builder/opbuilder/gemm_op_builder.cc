// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class GemmOpBuilder : public BaseOpBuilder {
 public:
  GemmOpBuilder() : BaseOpBuilder("GemmOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GemmOpBuilder);

 protected:
  Ort::Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                            const OrtNodeUnit& node_unit,
                            const Ort::Logger& logger,
                            std::vector<std::string>& input_names,
                            bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Ort::Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                          const OrtNodeUnit& node_unit,
                                          std::vector<std::string>&& input_names,
                                          const Ort::Logger& logger,
                                          bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Ort::Status ExplictOpCheck(const OrtNodeUnit& node_unit) const;
};

Ort::Status GemmOpBuilder::ExplictOpCheck(const OrtNodeUnit& node_unit) const {
  OrtNodeAttrHelper node_helper(node_unit);
  auto alpha = node_helper.Get("alpha", (float)1.0);
  RETURN_IF(alpha != 1.0, "QNN FullyConnected Op only support alpha=1.0.");
  auto beta = node_helper.Get("beta", (float)1.0);
  RETURN_IF(beta != 1.0, "QNN FullyConnected Op only support beta=1.0.");

  // input C shape need to be [M] or [1, M]
  if (node_unit.Inputs().size() == 3) {
    auto& inputB = node_unit.Inputs()[1];
    std::vector<uint32_t> inputB_shape;
    QnnModelWrapper::GetOnnxShape(inputB.shape, inputB_shape);

    auto& inputC = node_unit.Inputs()[2];
    std::vector<uint32_t> inputC_shape;
    QnnModelWrapper::GetOnnxShape(inputC.shape, inputC_shape);

    auto transB = node_helper.Get("transB", static_cast<int64_t>(0));
    auto M = (transB == 0) ? inputB_shape.at(1) : inputB_shape.at(0);
    RETURN_IF(inputC_shape.size() == 0 || (inputC_shape.size() == 1 && inputC_shape.at(0) != M) ||
                  (inputC_shape.size() == 2 && inputC_shape.at(1) != M),
              "QNN FullyConnected Op only support C with shape [N, M].");

    RETURN_IF(inputC_shape.size() == 2 && node_unit.Inputs()[2].quant_param.has_value() && inputC_shape.at(0) != 1,
              "QNN FullyConnected Op only support quantized C with shape [1, M].");
  }

  return Ort::Status();
}

Ort::Status GemmOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                         const OrtNodeUnit& node_unit,
                                         const Ort::Logger& logger,
                                         std::vector<std::string>& input_names,
                                         bool do_op_validation) const {
  if (do_op_validation) {
    RETURN_IF_ERROR(ExplictOpCheck(node_unit));
  }
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  // for Input A, B, C: 1 -- need transpose, 0 -- not needed
  std::vector<int64_t> input_trans_flag(3, 0);
  OrtNodeAttrHelper node_helper(node_unit);
  input_trans_flag.at(0) = node_helper.Get("transA", (int64_t)0);
  auto transB = node_helper.Get("transB", (int64_t)0);
  // QNN input_1 [m, n] vs Onnx [n, m]
  input_trans_flag.at(1) = transB == 0 ? 1 : 0;

  const auto& inputs = node_unit.Inputs();
  for (size_t input_i = 0; input_i < inputs.size(); ++input_i) {
    QnnQuantParamsWrapper quantize_param;
    RETURN_IF_ERROR(quantize_param.Init(qnn_model_wrapper, inputs[input_i]));

    bool is_quantized_tensor = inputs[input_i].quant_param.has_value();
    const auto& input_name = inputs[input_i].name;

    // Only skip if the input tensor has already been added (by producer op) *and* we don't need
    // to transpose it.
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name) && input_trans_flag[input_i] == 0) {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_name).c_str());
      input_names.push_back(input_name);
      continue;
    }

    ONNXTensorElementDataType input_type = inputs[input_i].type;
    RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, input_type, qnn_data_type));

    std::vector<uint32_t> input_shape;
    RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].shape, input_shape), "Cannot get shape");

    std::vector<uint8_t> unpacked_tensor;
    bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
    if (is_constant_input) {
      const auto* input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      if (1 == input_trans_flag.at(input_i)) {
        RETURN_IF_ERROR(quantize_param.HandleTranspose<size_t>(std::vector<size_t>({1, 0})));
        RETURN_IF_ERROR(
            utils::TwoDimensionTranspose(qnn_model_wrapper, input_shape, input_tensor, unpacked_tensor, logger));
      } else {
        RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(input_tensor, unpacked_tensor));
      }
    }

    std::string input_tensor_name = input_name;
    if (1 == input_trans_flag.at(input_i) && !is_constant_input) {
      RETURN_IF(quantize_param.IsPerChannel(), "Non-constant Gemm inputs only support per-tensor quantization");

      // Add Transpose node
      std::vector<uint32_t> old_input_shape(input_shape);
      input_shape[0] = old_input_shape[1];
      input_shape[1] = old_input_shape[0];
      const std::string& node_input_name(input_name);
      input_tensor_name = utils::UniqueNameGenerator().New(input_tensor_name, "_transpose");
      std::vector<uint32_t> perm{1, 0};
      RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), node_input_name, input_tensor_name,
                                                         old_input_shape, perm, input_shape,
                                                         qnn_data_type, quantize_param, do_op_validation,
                                                         qnn_model_wrapper.IsGraphInput(node_input_name)));
    }

    // Reshape [1, M] shape Bias.
    if (2 == input_i && 2 == input_shape.size() && input_shape[0] == 1) {
      input_shape[0] = input_shape[1];
      input_shape.resize(1);
    }

    input_names.push_back(input_tensor_name);
    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_tensor_name);
    QnnTensorWrapper input_tensorwrapper(input_tensor_name, tensor_type, qnn_data_type, std::move(quantize_param),
                                         std::move(input_shape), std::move(unpacked_tensor));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  // Insert QNN Convert op when activation input quant type differs from output quant type
  // HTP FullyConnected requires in[0] and out[0] to have the same quantized data type
  // E.g., for U8 activation with U16 output, insert Convert(U8->U16) before FullyConnected
  if (!input_names.empty() && !node_unit.Outputs().empty()) {
    TensorInfo input_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

    const bool has_input_quant = node_unit.Inputs()[0].quant_param.has_value();
    const bool has_output_quant = node_unit.Outputs()[0].quant_param.has_value();
    const bool has_quant_weights = node_unit.Inputs().size() >= 2 &&
                                   node_unit.Inputs()[1].quant_param.has_value();
    const bool is_mixed_precision = input_info.qnn_data_type != output_info.qnn_data_type;

    // Sub-cases of mixed precision
    const bool is_widening = is_mixed_precision && has_input_quant && has_output_quant &&
                             utils::GetElementSizeByType(input_info.qnn_data_type) <
                                 utils::GetElementSizeByType(output_info.qnn_data_type);
    const bool is_quant_to_float = is_mixed_precision && has_input_quant && !has_output_quant &&
                                   output_info.qnn_data_type == QNN_DATATYPE_FLOAT_32;

    // FC must run in fp32 mode when activation is dequantized or already fp32 with quant weights
    const bool needs_fp32_fc = is_quant_to_float ||
                               (input_info.qnn_data_type == QNN_DATATYPE_FLOAT_32 && has_quant_weights);

    if (is_widening) {
      RETURN_IF_NOT(input_info.quant_param.IsPerTensor(),
                    "Mixed-precision Gemm activation input only supports per-tensor quantization");
      const Qnn_QuantizeParams_t& input_qp = input_info.quant_param.Get();
      float output_scale = input_qp.scaleOffsetEncoding.scale / 256.0f;
      int32_t output_offset = input_qp.scaleOffsetEncoding.offset * 256;

      const std::string convert_output_name =
          utils::UniqueNameGenerator().New(input_names[0], "_convert_to_output_type");
      QnnTensorWrapper convert_output_tensorwrapper(convert_output_name,
                                                    QNN_TENSOR_TYPE_NATIVE,
                                                    output_info.qnn_data_type,
                                                    QnnQuantParamsWrapper(output_scale, output_offset),
                                                    std::move(input_info.shape));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(convert_output_tensorwrapper)),
                    "Failed to add Convert output tensor");
      RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::UniqueNameGenerator().New(convert_output_name, QNN_OP_CONVERT),
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_CONVERT,
                        {input_names[0]},
                        {convert_output_name},
                        {},
                        do_op_validation),
                    "Failed to add Convert node");
      input_names[0] = convert_output_name;

      // Requantize bias for the new activation scale. The bias was quantized with
      // bias_scale = activation_scale * weight_scale. After Convert, the activation scale
      // changed (e.g., /256 for U8->U16), so the bias must be requantized to match
      if (input_names.size() == 3) {
        const auto& bias_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[2]);
        if (bias_wrapper.GetTensorDataType() == QNN_DATATYPE_SFIXED_POINT_32) {
          const auto& bias_qp = bias_wrapper.GetQnnQuantParams();
          const auto& bias_dims = bias_wrapper.GetTensorDims();

          std::vector<float> bias_scales;
          RETURN_IF_ERROR(bias_qp.GetScales(bias_scales));
          std::vector<int32_t> bias_offsets(bias_scales.size(), 0);

          TensorInfo weight_info = {};
          RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], weight_info));
          std::vector<float> weight_scales;
          RETURN_IF_ERROR(weight_info.quant_param.GetScales(weight_scales));

          const Qnn_Tensor_t& qnn_bias = bias_wrapper.GetQnnTensor();
          const Qnn_ClientBuffer_t& client_buf = GetQnnTensorClientBuf(qnn_bias);
          std::vector<uint8_t> original_bias_data(
              static_cast<const uint8_t*>(client_buf.data),
              static_cast<const uint8_t*>(client_buf.data) + client_buf.dataSize);

          std::vector<uint8_t> requantized_bias_data;
          std::vector<float> new_bias_scales;
          std::vector<int32_t> new_bias_offsets;
          std::optional<int64_t> axis = bias_qp.IsPerChannel()
                                            ? std::optional<int64_t>(0)
                                            : std::nullopt;

          RETURN_IF_ERROR(utils::RequantizeBiasTensor(
              original_bias_data, bias_dims,
              bias_scales, bias_offsets,
              weight_scales, output_scale,
              QNN_DATATYPE_SFIXED_POINT_32,
              requantized_bias_data, new_bias_scales, new_bias_offsets,
              axis));

          const std::string new_bias_name =
              utils::UniqueNameGenerator().New(input_names[2], "_requant");
          QnnQuantParamsWrapper new_bias_qp;
          if (bias_qp.IsPerChannel()) {
            new_bias_qp = QnnQuantParamsWrapper(new_bias_scales, new_bias_offsets,
                                                /*axis*/ 0, /*is_int4*/ false);
          } else {
            new_bias_qp = QnnQuantParamsWrapper(new_bias_scales[0], new_bias_offsets[0]);
          }
          QnnTensorWrapper new_bias_wrapper(new_bias_name, QNN_TENSOR_TYPE_STATIC,
                                            QNN_DATATYPE_SFIXED_POINT_32,
                                            std::move(new_bias_qp),
                                            std::vector<uint32_t>(bias_dims),
                                            std::move(requantized_bias_data));
          RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(new_bias_wrapper)),
                        "Failed to add requantized bias tensor.");
          input_names[2] = new_bias_name;
        }
      }
    }

    if (needs_fp32_fc) {
      // Precompute fp32 weights from quantized weights so FC runs in float mode
      if (input_names.size() >= 2 && has_quant_weights) {
        TensorInfo weight_info = {};
        RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], weight_info));

        const auto& weight_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(input_names[1]);
        const Qnn_Tensor_t& qnn_weight = weight_wrapper.GetQnnTensor();
        const Qnn_ClientBuffer_t& w_buf = GetQnnTensorClientBuf(qnn_weight);

        // Use wrapper dims (post-transpose) and wrapper quant params (already transposed)
        const auto& w_dims = weight_wrapper.GetTensorDims();
        const auto& w_qp = weight_wrapper.GetQnnQuantParams();
        std::vector<float> weight_scales;
        RETURN_IF_ERROR(w_qp.GetScales(weight_scales));
        std::vector<int32_t> weight_offsets(weight_scales.size(), 0);

        size_t num_elements = 1;
        for (auto d : w_dims) num_elements *= d;

        std::vector<float> fp32_weights(num_elements);
        std::optional<int64_t> axis = w_qp.IsPerChannel() ? std::optional<int64_t>(0) : std::nullopt;
        RETURN_IF_ERROR(utils::DequantizePerChannel(
            gsl::span<const uint8_t>(static_cast<const uint8_t*>(w_buf.data), w_buf.dataSize),
            gsl::span<const uint32_t>(w_dims.data(), w_dims.size()),
            weight_scales, weight_offsets,
            fp32_weights, weight_info.qnn_data_type, axis));

        std::vector<uint8_t> fp32_bytes(num_elements * sizeof(float));
        memcpy(fp32_bytes.data(), fp32_weights.data(), fp32_bytes.size());

        const std::string fp32_weight_name =
            utils::UniqueNameGenerator().New(input_names[1], "_fp32");
        QnnTensorWrapper fp32_weight_tensor(fp32_weight_name, QNN_TENSOR_TYPE_STATIC,
                                            QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                            std::vector<uint32_t>(w_dims.begin(), w_dims.end()),
                                            std::move(fp32_bytes));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fp32_weight_tensor)),
                      "Failed to add fp32 weight tensor.");
        input_names[1] = fp32_weight_name;
      }

      // Insert QNN Dequantize op for activation (input[0]): per-tensor quantized → fp32
      if (is_quant_to_float) {
        const std::string dq_output_name =
            utils::UniqueNameGenerator().New(input_names[0], "_dequantize");
        std::vector<uint32_t> dq_shape = input_info.shape;
        QnnTensorWrapper dq_output_tensor(dq_output_name, QNN_TENSOR_TYPE_NATIVE,
                                          QNN_DATATYPE_FLOAT_32, QnnQuantParamsWrapper(),
                                          std::move(dq_shape));
        RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(dq_output_tensor)),
                      "Failed to add Dequantize output tensor for activation.");
        RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::UniqueNameGenerator().New(dq_output_name, QNN_OP_DEQUANTIZE),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_DEQUANTIZE,
                          {input_names[0]},
                          {dq_output_name},
                          {},
                          do_op_validation),
                      "Failed to add Dequantize node for activation.");
        input_names[0] = dq_output_name;
      }
    }
  }

  return Ort::Status();
}

Ort::Status GemmOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                       const OrtNodeUnit& node_unit,
                                                       std::vector<std::string>&& input_names,
                                                       const Ort::Logger& logger,
                                                       bool do_op_validation) const {
  // Decompose Gemm into FullyConnected + Add when:
  // 1. Bias (input C) has 2D shape [N, M] where N != 1 (FC doesn't support this shape), OR
  // 2. Bias is an intermediate (NATIVE) tensor produced by another op (QNN FC requires static bias).
  bool split_gemm = false;
  if (node_unit.Inputs().size() == 3) {
    auto& input_c = node_unit.Inputs()[2];
    std::vector<uint32_t> input_c_shape;
    QnnModelWrapper::GetOnnxShape(input_c.shape, input_c_shape);

    // Split when input_c has 2d shape and not [1, M]
    split_gemm = (input_c_shape.size() == 2 && input_c_shape.at(0) != 1);

    // Split when bias is an intermediate (NATIVE) tensor produced by another op.
    // ORT's MatMulAddFusion can fuse MatMul+Add->Gemm where the Add's other input
    // is an intermediate tensor (e.g., output of another MatMul). QNN FC requires
    // bias to be either STATIC (constant) or APP_WRITE (graph input), not NATIVE.
    split_gemm = split_gemm || qnn_model_wrapper.GetTensorType(input_c.name) == QNN_TENSOR_TYPE_NATIVE;
  }

  TensorInfo act_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], act_info));
  TensorInfo out_info = {};
  RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], out_info));

  const bool has_input_quant = node_unit.Inputs()[0].quant_param.has_value();
  const bool has_output_quant = node_unit.Outputs()[0].quant_param.has_value();
  const bool is_mixed_precision = act_info.qnn_data_type != out_info.qnn_data_type;

  // Sub-cases: narrowing (e.g., U16->U8), float-to-quant (fp32->U8/U16)
  const bool is_narrowing = is_mixed_precision && has_input_quant && has_output_quant &&
                            utils::GetElementSizeByType(act_info.qnn_data_type) >
                                utils::GetElementSizeByType(out_info.qnn_data_type);
  const bool is_float_to_quant = is_mixed_precision && !has_input_quant && has_output_quant &&
                                 act_info.qnn_data_type == QNN_DATATYPE_FLOAT_32;

  const std::string& org_output_name = node_unit.Outputs()[0].name;
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);

  // For narrowing/float-to-quant, FC output uses intermediate name; post-processing follows
  const std::string fc_chain_output_name = (is_narrowing || is_float_to_quant)
                                               ? utils::UniqueNameGenerator().New(org_output_name, "_pre_convert")
                                               : org_output_name;

  QnnQuantParamsWrapper narrowing_fc_qp;
  float narrowing_act_scale = 0.0f;
  if (is_narrowing) {
    RETURN_IF_NOT(out_info.quant_param.IsPerTensor(),
                  "Mixed-precision Gemm output only supports per-tensor quantization");
    const Qnn_QuantizeParams_t& out_qp = out_info.quant_param.Get();
    narrowing_act_scale = out_qp.scaleOffsetEncoding.scale / 256.0f;
    int32_t narrowing_offset = out_qp.scaleOffsetEncoding.offset * 256;
    narrowing_fc_qp = QnnQuantParamsWrapper(narrowing_act_scale, narrowing_offset);

    // No bias requantization needed for narrowing: FC accumulates at input_act_scale * weight_scale,
    // which matches the original bias scale from weight_bias_quantization. The narrowing_fc_qp only
    // controls how FC quantizes its accumulated output, not the accumulation itself.
  }

  if (split_gemm) {
    std::vector<uint32_t> output_shape = out_info.shape;

    // Create FullyConnected Node
    std::vector<std::string> gemm_input_0_1;
    gemm_input_0_1.push_back(input_names[0]);
    gemm_input_0_1.push_back(input_names[1]);
    const std::string fc_output_name = utils::UniqueNameGenerator().New(org_output_name, "_fc");
    QnnTensorWrapper fully_connected_output(fc_output_name, QNN_TENSOR_TYPE_NATIVE, act_info.qnn_data_type,
                                            QnnQuantParamsWrapper(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fully_connected_output)),
                  "Failed to add FullyConnected output tensor.");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_FULLY_CONNECTED),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_FULLY_CONNECTED,
                                                  std::move(gemm_input_0_1),
                                                  {fc_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add FullyConnected node.");

    // Create Add Node — uses derived quant params when narrowing
    Qnn_DataType_t add_out_type = (is_narrowing || is_float_to_quant) ? act_info.qnn_data_type : out_info.qnn_data_type;
    QnnQuantParamsWrapper add_out_qp = is_narrowing        ? narrowing_fc_qp.Copy()
                                       : is_float_to_quant ? QnnQuantParamsWrapper()
                                                           : out_info.quant_param.Copy();
    Qnn_TensorType_t add_tensor_type = (!(is_narrowing || is_float_to_quant) && is_graph_output)
                                           ? QNN_TENSOR_TYPE_APP_READ
                                           : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper add_output_tensor(fc_chain_output_name, add_tensor_type, add_out_type,
                                       std::move(add_out_qp), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(add_output_tensor)),
                  "Failed to add ElementWiseAdd output tensor");

    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_ELEMENT_WISE_ADD),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_ELEMENT_WISE_ADD,
                                                  {fc_output_name, input_names[2]},
                                                  {fc_chain_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add ElementWiseAdd node.");
  } else if (is_narrowing) {
    // Non-split path with narrowing: FC outputs input activation type, derived quant params
    std::vector<uint32_t> output_shape = out_info.shape;
    QnnTensorWrapper fc_output_tensor(fc_chain_output_name, QNN_TENSOR_TYPE_NATIVE, act_info.qnn_data_type,
                                      narrowing_fc_qp.Copy(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fc_output_tensor)),
                  "Failed to add FullyConnected output tensor");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_FULLY_CONNECTED),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_FULLY_CONNECTED,
                                                  std::move(input_names),
                                                  {fc_chain_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add FullyConnected node");
  } else if (is_float_to_quant) {
    // Non-split path with float-to-quant: FC outputs fp32, Quantize follows
    std::vector<uint32_t> output_shape = out_info.shape;
    QnnTensorWrapper fc_output_tensor(fc_chain_output_name, QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_32,
                                      QnnQuantParamsWrapper(), std::vector<uint32_t>(output_shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(fc_output_tensor)),
                  "Failed to add FullyConnected output tensor");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::UniqueNameGenerator().New(node_unit, QNN_OP_FULLY_CONNECTED),
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_FULLY_CONNECTED,
                                                  std::move(input_names),
                                                  {fc_chain_output_name},
                                                  {},
                                                  do_op_validation),
                  "Failed to add FullyConnected node");
  } else {
    RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), {},
                                   logger, do_op_validation, GetQnnOpType(node_unit.OpType())));
  }

  // Insert Convert after FC chain for narrowing (e.g., U16->U8)
  if (is_narrowing) {
    Qnn_TensorType_t final_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper final_output_tensor(org_output_name, final_tensor_type, out_info.qnn_data_type,
                                         out_info.quant_param.Copy(), std::vector<uint32_t>(out_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)),
                  "Failed to add Convert output tensor");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(org_output_name, QNN_OP_CONVERT),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_CONVERT,
                      {fc_chain_output_name},
                      {org_output_name},
                      {},
                      do_op_validation),
                  "Failed to add Convert node");
  }

  // Insert Quantize after FC chain for float-to-quant (fp32 -> quantized output)
  if (is_float_to_quant) {
    Qnn_TensorType_t final_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper final_output_tensor(org_output_name, final_tensor_type, out_info.qnn_data_type,
                                         out_info.quant_param.Copy(), std::vector<uint32_t>(out_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(final_output_tensor)),
                  "Failed to add Quantize output tensor");
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                      utils::UniqueNameGenerator().New(org_output_name, QNN_OP_QUANTIZE),
                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                      QNN_OP_QUANTIZE,
                      {fc_chain_output_name},
                      {org_output_name},
                      {},
                      do_op_validation),
                  "Failed to add Quantize node");
  }

  return Ort::Status();
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GemmOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
