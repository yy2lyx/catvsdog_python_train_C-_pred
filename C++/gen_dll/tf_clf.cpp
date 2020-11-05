#include "pch.h"
#include "tf_clf.h"


Mat TFClf::preprocess_img(Mat img) {
    //图像进行resize处理
    resize(img, img, cv::Size(resize_col, resize_row));

    //去均值
    img.convertTo(img, CV_32FC3);
    // 遍历所有的像素的每个通道，减去对应的均值（BGR这种顺序）
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            img.at<Vec3f>(row, col)[0] -= mean[0];
            img.at<Vec3f>(row, col)[1] -= mean[1];
            img.at<Vec3f>(row, col)[2] -= mean[2];
        }
    }
    return img;
}

void TFClf::mat_to_tensor(Mat img, Tensor* output_tensor) {
    //创建一个指向tensor的内容的指针
    float* p = output_tensor->flat<float>().data();

    //创建一个Mat，与tensor的指针绑定,改变这个Mat的值，就相当于改变tensor的值
    cv::Mat tempMat(resize_row, resize_col, CV_32FC3, p);
    img.convertTo(tempMat, CV_32FC3);
}

void TFClf::show_result_pic(Mat img, int output_class_id,double output_prob) {
    string rs_string;
    if (output_class_id == 0) {
        rs_string = "Class:Dog Proba:" + to_string(output_prob);
    }
    else {
        rs_string = "Class:Cat Proba:" + to_string(output_prob);
    }
    // 把label输出到图片中
    cv::putText(img, rs_string, draw_point, FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
    cv::imshow("result", img);
    cv::waitKey(0);
}

void TFClf::model_pred() {
    /*--------------------------------创建session------------------------------*/
    Session* session;
    Status status = NewSession(SessionOptions(), &session);//创建新会话Session

    /*--------------------------------从pb文件中读取模型--------------------------------*/
    GraphDef graphdef; //Graph Definition for current model

    Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef); //从pb文件中读取图模型;
    if (!status_load.ok()) {
        std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
        std::cout << status_load.ToString() << "\n";
    }
    Status status_create = session->Create(graphdef); //将模型导入会话Session中;
    if (!status_create.ok()) {
        std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
    }
    std::cout << "<----Successfully created session and load graph.------->" << endl;

    /*---------------------------------载入测试图片-------------------------------------*/
    std::cout << endl << "<------------loading test_image-------------->" << endl;
    Mat img = imread(image_path);


    if (img.empty())
    {
        std::cout << "can't open the image!!!!!!!" << endl;
    }
    // 图像预处理
    Mat resize_img;
    resize_img = preprocess_img(img);
    //创建一个tensor作为输入网络的接口
    Tensor resized_tensor(DT_FLOAT, TensorShape({ 1,resize_row,resize_col,3 }));

    //将Opencv的Mat格式的图片存入tensor
    mat_to_tensor(resize_img, &resized_tensor);

    /*-----------------------------------用网络进行测试-----------------------------------------*/
    std::cout << endl << "<-------------Running the model with test_image--------------->" << endl;
    //前向运行，输出结果一定是一个tensor的vector
    vector<tensorflow::Tensor> outputs;
    string output_node = output_tensor_name;
    Status status_run = session->Run({ {input_tensor_name, resized_tensor} }, { output_node }, {}, &outputs);

    if (!status_run.ok()) {
        std::cout << "ERROR: RUN failed..." << std::endl;
        std::cout << status_run.ToString() << "\n";
    }
    //把输出值给提取出来
    Tensor t = outputs[0];                   // Fetch the first tensor
    auto tmap = t.tensor<float, 2>();        // Tensor Shape: [batch_size, target_class_num]
    int output_dim = t.shape().dim_size(1);  // Get the target_class_num from 1st dimension
    // Argmax: Get Final Prediction Label and Probability
    int output_class_id = -1;
    double output_prob = 0.0;
    for (int j = 0; j < output_dim; j++)
    {
        std::cout << "Class " << j << " prob:" << tmap(0, j) << "," << std::endl;
        if (tmap(0, j) >= output_prob) {
            output_class_id = j;
            output_prob = tmap(0, j);
        }
    }
    // 展示图片结果
    show_result_pic(img, output_class_id, output_prob);

}
