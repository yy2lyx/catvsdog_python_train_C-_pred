from keras.models import load_model
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.framework import graph_io
# 针对tf2.x来说不支持freezegraph的，这里需要使用tf1的方式
tf.compat.v1.disable_eager_execution()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


"""----------------------------------配置路径-----------------------------------"""
h5_model_path = 'model/model.h5'
pb_model_name = 'model.pt'
output_path = '.'

"""----------------------------------导入keras模型------------------------------"""
K.set_learning_phase(0)
net_model = load_model(h5_model_path)

print('input is :', net_model.input.name)
print('output is:', net_model.output.name)

"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in net_model.outputs])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)