

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#define NDEBUG
#include <assert.h>
#define assert(ignore) ((void)0)

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

            nvinfer1::ITensor* boxes_tensor = args[0].ITensor();
            nvinfer1::ITensor* scores_tensor = args[1].ITensor();
            auto scores_tensor = util::toVec(scores_tesnor->getDimensions()); 
            auto nms_thresh = args[2].unwrapToDouble();
             
            // concatenate additional tesnor to boxes with the indices
            
            // expand dims for boxes tensor
            auto shuffle_layer1 = ctx->net->addShuffle(*boxes_tensor);
            TRTORCH_CHECK(shuffle_layer1, "Unable to create shuffle layer from node: " << *n);
            shuffle_layer->setReshapeDimensions(util::unsqueezeDims(boxes_tensor->getDimensions(), 1));
            auto boxes_tensor = shuffle_layer1->getOutput(0)
                
            auto shuffle_layer2 = ctx->net->addShuffle(*boxes_tensor);
            TRTORCH_CHECK(shuffle_layer2, "Unable to create shuffle layer from node: " << *n);
            shuffle_layer->setReshapeDimensions(util::unsqueezeDims(boxes_tensor->getDimensions(), 0));
            auto boxes_tensor = shuffle_layer2->getOutput(0)
            
            //expand dims for scores tensor
            auto shuffle_layer3 = ctx->net->addShuffle(*scores_tensor);
            TRTORCH_CHECK(shuffle_layer3, "Unable to create shuffle layer from node: " << *n);
            shuffle_layer->setReshapeDimensions(util::unsqueezeDims(boxes_tensor->getDimensions(), 0));
            auto scores_tensor = shuffle_layer3->getOutput(0)
                
            
            

            //const int numInputs = inputs.size();
            //LOG_ERROR("no of inputs are "<<numInputs);
            //LOG_ERROR("node outsize and op type are "<<node.output().size()<< " type " << node.op_type());

            const auto scores_dims = scores_tensor->getDimensions();
            const auto boxes_dims = boxes_tensor->getDimensions();
            LOG_DEBUG("boxes dims "<< boxes_dims.nbDims << " dim 3 has size "<<boxes_dims.d[2]);
            
            
            
            bool share_location = true;
            const bool is_normalized = true;
            const bool clip_boxes = true;
            int backgroundLabelId = 0;
            // Initialize.
            
            
            
            nvinfer1::plugin::NMSParameters params{};
            params.shareLocation =  true;
            params.backgroundLabelId =  -1;
            params.numClasses=1;
            params.topK = 1000;
            params.keepTopK = 100;
            params.scoreThreshold = 0.7f;
            params.iouThreshold =  nms_thresh;
            //params.clipBoxes = true;
            params.isNormalized =  true;
            
            std::vector<nvinfer1::PluginField> f;
            f.emplace_back("shareLocation", &share_location, nvinfer1::PluginFieldType::kINT32, 1);
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
            
            nvinfer1::PluginFieldCollection fc;
            fc.nbFields = f.size();
            fc.fields = f.data();
            
//             auto creator = getPluginRegistry()->getPluginCreator("BatchedNMS_TRT", "1");
//             auto plugin = creator->createPlugin("NMSPlugintrtorch", &fc);   
//             LOG_DEBUG("plugin type " << plugin->getPluginType());
            
            auto plugin = createBatchedNMSPlugin(params);
            LOG_DEBUG("PLUGIN CREATED");
            std::vector<nvinfer1::ITensor*> in = {boxes_tensor, scores_tensor};
//             auto nms_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 2, *plugin); 
            auto nms_layer = ctx->net->addPluginV2(in.data(), in.size(), *plugin); 
            LOG_DEBUG("test1");
            TRTORCH_CHECK(nms_layer, "Unable to create layer for torchvision::nms");
            
            nms_layer->setName(util::node_info(n).c_str());
            
            auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], nms_layer->getOutput(0));
            LOG_DEBUG("test2");
            //LOG_DEBUG("NMS layer output tensor shape: " << layer_output->getDimensions());
            LOG_DEBUG("test3");
            return true;
        }
    });
    
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch