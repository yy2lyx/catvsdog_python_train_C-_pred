import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import numpy as np

def pred_with_h5(processed_img):
    model = load_model("model/model.h5")
    rs = model.predict(processed_img)
    return rs


def pred_with_pt(img, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.compat.v1.GraphDef()

        # 打开.pb模型
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print("tensors:", tensors)

        # 在一个session中去run一个前向
        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            op = sess.graph.get_operations()

            input_x = sess.graph.get_tensor_by_name("conv2d_input:0")  # 具体名称看上一段代码的input.name
            print("input_X:", input_x)

            out_softmax = sess.graph.get_tensor_by_name("dense/Softmax:0")  # 具体名称看上一段代码的output.name
            print("Output:", out_softmax)

            img_out_softmax = sess.run(out_softmax,
                                       feed_dict={input_x: img})

            return img_out_softmax


if __name__ == '__main__':
    img = cv2.imread("data/test_set/test_set/cats/cat.4001.jpg")
    img = cv2.resize(img,(224,224))
    print(img.shape)
    # cv2.imshow("src",img)

    # 取均值化：移除图像的平均亮度值，去除图像的光照亮度干扰
    test = np.mean(img, axis=0)
    rs_img = img - test
    cv2.imshow("1",rs_img)

    # 预处理
    processed_img = preprocess_input(img)
    cv2.imshow("preprocess",processed_img)
    cv2.waitKey(0)
    processed_img = np.expand_dims(processed_img,axis=0)
    rs_img = np.expand_dims(rs_img,axis=0)




    print(processed_img.shape)


    # load pt model
    rs_pt = pred_with_pt(processed_img, 'model/model.pt')[0][0]
    rs_h5 = pred_with_h5(processed_img)[0][0]

    if rs_pt == rs_h5:
        print("The same!")
    else:
        print("No same!")


