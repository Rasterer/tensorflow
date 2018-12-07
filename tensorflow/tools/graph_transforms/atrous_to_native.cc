/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status AtrousToNative(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"BatchToSpaceND",
          {
              {"Conv2D|DepthwiseConv2dNative",
                  {
                      {"SpaceToBatchND",
                          {
                              {"*"},          // Input to the flattened op.
                              {"*"},          // block_shape
                              {"*"}           // paddings
                          }
                      },
                      {"*"}                   // filter
                  }
              },
              {"*"},                          // block_shape
              {"*"}                           // crops
          }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& batch_to_space_node = match.node;
        const NodeDef& conv_node = match.inputs[0].node;
        const NodeDef& filter_node = match.inputs[0].inputs[1].node;
        const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;
        const NodeDef& space_to_batch_block_shape_node =
            match.inputs[0].inputs[0].inputs[1].node;

        // The atrous rate value is inferred from the block shape.
        Tensor block_shape =
            GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
        const int32 block_height = block_shape.flat<int32>()(0);
        const int32 block_width = block_shape.flat<int32>()(1);

        NodeDef conv_native_node;

        conv_native_node.set_name(batch_to_space_node.name());
        conv_native_node.set_op(conv_node.op());
        conv_native_node.set_device(conv_node.device());

        AddNodeInput(input_node.name(), &conv_native_node);
        AddNodeInput(filter_node.name(), &conv_native_node);

        CopyNodeAttr(conv_node, "T", "T", &conv_native_node);
        CopyNodeAttr(conv_node, "strides", "strides", &conv_native_node);
        SetNodeAttr("padding", "SAME", &conv_native_node);
        SetNodeAttr("dilations", gtl::ArraySlice<int32>({1, block_height, block_width, 1}), &conv_native_node);
        CopyNodeAttr(conv_node, "data_format", "data_format",
                     &conv_native_node);

        if (conv_node.op() == "Conv2D") {
          CopyNodeAttr(conv_node, "use_cudnn_on_gpu", "use_cudnn_on_gpu",
                       &conv_native_node);
        }

        new_nodes->push_back(input_node);
        new_nodes->push_back(filter_node);
        new_nodes->push_back(conv_native_node);

        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("atrous_to_native", AtrousToNative);

}  // namespace graph_transforms
}  // namespace tensorflow
