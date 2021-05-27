#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto nms TRTORCH_UNUSED = RegisterNodeConversionPatterns()
    .pattern({
        "torchvision::nms(Tensor boxes_tensor, Tensor scores_tensor, Scalar iou_threshold) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            // NonMaxSuppression is not supported opset below 10.
            //ASSERT(ctx->getOpsetVersion() >= 10, ErrorCode::kUNSUPPORTED_NODE);

            auto boxes_tensor = args[0].ITensor();
            auto scores_tensor = args[1].ITensor();
            auto nms_thresh = args[2].unwrapToDouble();

            //const int numInputs = inputs.size();
            //LOG_ERROR("no of inputs are "<<numInputs);
            //LOG_ERROR("node outsize and op type are "<<node.output().size()<< " type " << node.op_type());

            const auto scores_dims = scores_tensor->getDimensions();
            const auto boxes_dims = boxes_tensor->getDimensions();
            LOG_ERROR("boxes dims "<< boxes_dims.nbDims << " dim3 has size "<<boxes_dims.d[2]);
            
            std::vector<nvinfer1::PluginField> f;
            nvinfer1::PluginFieldCollection fc;
            bool share_location = true;
            const bool is_normalized = true;
            const bool clip_boxes = true;
            int backgroundLabelId = 0;
            // Initialize.
            
            nvinfer1::plugin::DetectionOutputParameters params{};
            params.shareLocation =  true;
            params.isNormalized =  true;
            params.backgroundLabelId =  0;
            params.nmsThreshold =  nms_thresh;
            
            
            f.emplace_back("shareLocation", &share_location, nvinfer1::PluginFieldType::kINT8, 1);
            f.emplace_back("isNormalized", &is_normalized, nvinfer1::PluginFieldType::kINT8, 1);
            f.emplace_back("clipBoxes", &clip_boxes, nvinfer1::PluginFieldType::kINT8, 1);
            f.emplace_back("backgroundLabelId", &backgroundLabelId, nvinfer1::PluginFieldType::kINT32, 1);
            f.emplace_back("nmsThreshold", &nms_thresh, nvinfer1::PluginFieldType::kFLOAT32, 1);
            // Create plugin from registry
            // nvinfer1::IPluginV2* plugin = importPluginFromRegistry(ctx, pluginName, pluginVersion, node.name(), f);
            //nvinfer1::IPluginV2* plugin = createPlugin(node.name(), importPluginCreator(pluginName, pluginVersion), f);
            //nvinfer1::IPluginV2* plugin = createNMSPlugin(params);
//             ASSERT(plugin != nullptr && "NonMaxSuppression plugin was not found in the plugin registry!",
//                        ErrorCode::kUNSUPPORTED_NODE);
            
            fc.nbFields = f.size();
            fc.fields = f.data();
            
            auto creator = getPluginRegistry()->getPluginCreator("NMS_TRT", "1");
            auto plugin = creator->createPlugin("NMSPlugintrtorch", &fc);
            
            std::vector<nvinfer1::ITensor*> in ={&boxes_tensor, &scores_tensor};
            auto nms_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);  
            TRTORCH_CHECK(nms_layer, "Unable to create layer for torchvision::nms");
            
            nms_layer->setName(util::node_info(n).c_str());
            
            auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], nms_layer->getOutput(0));

            LOG_DEBUG("NMS layer output tensor shape: " << layer_output->getDimensions());
            return true;
        }
    });
    
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch