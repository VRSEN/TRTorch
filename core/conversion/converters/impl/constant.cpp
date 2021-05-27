#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto constant_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"trt::const(Tensor self) -> Tensor",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This converter may be abusing what the registry is supposed to be
              // used for Fundimentally this is because of the differing
              // philosophies between TensorRT and PyTorch, i.e. Variables contain
              // Tensors vs. just Tensors

              auto t = args[0].unwrapToTensor();
              auto t_weights = Weights(ctx, t);
              auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
              const_layer->setName(util::node_info(n).c_str());
              auto const_out = ctx->AssociateValueAndTensor(n->outputs()[0], const_layer->getOutput(0));

              LOG_DEBUG("Output tensor shape: " << const_out->getDimensions());

              return true;
            }})
    .pattern({"aten::Int.Tensor(Tensor a) -> (int)", //-torchscript схема функции для конвертации
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool { // ctx - struct в trtorch который содержит конфиг и nvinfer1::INetworkDefinition - класс определяющий структуру всей сетки и методы помошники
              auto t = args[0].ITensorOrFreeze(ctx);
              t->setType(nvinfer1::DataType::kINT32); //-1 аргумент ITensor! (не IValue)
                
              //конвертировать ITensor в int тут
                
              t = ctx->AssociateValueAndTensor(n->outputs()[0], t); //-привязать ITensor к outputу ноды (надо как то привязать int)
              
              //LOG_DEBUG("Output tensor shape: ");// << t->getDimensions());
              return true;
            }})
    ;
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
