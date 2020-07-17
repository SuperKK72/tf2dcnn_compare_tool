import os
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import time
import sys
import tensorflow as tf

network_config_file_path = './tf_inference_config_debug/' + sys.argv[1]
network_config_file = open(network_config_file_path)
network_config_parameters = []
for network_config_parameter in network_config_file:
    if len(network_config_parameter.strip()) != 0:
        # print(network_config_parameter)
        network_config_parameter = network_config_parameter[network_config_parameter.find('=')+1:].strip()
        if network_config_parameter[0] != '#':
            network_config_parameters.append(network_config_parameter)
network_config_file.close()

net_name    = network_config_parameters[0]
net_path    = network_config_parameters[1]
input_name  = network_config_parameters[2]
output_name = network_config_parameters[3]
img_path    = network_config_parameters[4]


img_n       = int(network_config_parameters[5])
img_c       = int(network_config_parameters[6])
img_h       = int(network_config_parameters[7])
img_w       = int(network_config_parameters[8])
std         = float(network_config_parameters[9])
val_B       = float(network_config_parameters[10])
val_G       = float(network_config_parameters[11])
val_R       = float(network_config_parameters[12])
val_D       = float(network_config_parameters[13])

rlt_path    = network_config_parameters[14]
debug_node_input_name  = network_config_parameters[15]
debug_node_output_name  = network_config_parameters[16]



def run_tf():

    with tf.Graph().as_default():

        graph_def = tf.GraphDef()
        with open(net_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input = sess.graph.get_tensor_by_name(input_name)
            output = sess.graph.get_tensor_by_name(output_name)
            debug_node_input = sess.graph.get_tensor_by_name(debug_node_input_name)
            debug_node_output = sess.graph.get_tensor_by_name(debug_node_output_name)

            image=cv2.imread(img_path, -1)
            image=cv2.resize(image, (img_w, img_h))
            # print(image)
            X=np.array(image).astype(np.float32)
            X[:, :, 0] = (X[:, :, 0] - val_B) * std
            X[:, :, 1] = (X[:, :, 1] - val_G) * std
            X[:, :, 2] = (X[:, :, 2] - val_R) * std
            if img_c == 4:
                X[:, :, 3] = (X[:, :, 3] - val_D) * std
            X=X.reshape((img_n, img_h, img_w, img_c))

            start = time.time()
            results = sess.run(output, feed_dict={input: X})
            debug_node_input_tensor = sess.run(debug_node_input, feed_dict={input: X})
            debug_node_output_tensor = sess.run(debug_node_output, feed_dict={input: X})


            end = time.time()
            # print(type(results))
            # print(results.shape)
            # print(len(results.shape))
            # print(results.shape[1])
            #print(results)
            #print(len(results.flatten()))

            print("net name: ", net_name)
            print("input size: ", X.shape)
            print("input name: ", input_name)
            print("output name: ", output_name)
            print("output shape: ", results.shape)
            print("runtime: ", end-start, "s")
            print("len of result: ", len(results.flatten()))
            print("max of result: ", max(results.flatten()))
            print("debug node input name: ", debug_node_input_name)
            print("debug node input shape: ", debug_node_input_tensor.shape)
            print("debug node output name: ", debug_node_output_name)
            print("debug node output shape: ", debug_node_output_tensor.shape)

if __name__ == "__main__":
    run_tf()