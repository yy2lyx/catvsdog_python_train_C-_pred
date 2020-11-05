#pragma once
#ifndef TF_CLF_H

#endif // !TF_CLF_H

#pragma comment(lib,"DllTF.lib")
class __declspec(dllexport) TFClf;

#include <fstream>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "opencv2/opencv.hpp"

using namespace tensorflow::ops;
using namespace tensorflow;
using namespace std;
using namespace cv;
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;



class TFClf {
private:
	vector<float> mean = { 103.939,116.779,123.68 };
	int resize_col = 224;
	int resize_row = 224;
	string input_tensor_name = "conv2d_input";
	string output_tensor_name = "dense/Softmax";
	Point draw_point = Point(50, 50);

public:
	string image_path, model_path;
	TFClf(string img, string model) :image_path(img), model_path(model) {}
	void mat_to_tensor(Mat img, Tensor* output_tensor);
	Mat preprocess_img(Mat img);
	void model_pred();
	void show_result_pic(Mat img, int output_class_id, double output_prob);
};
