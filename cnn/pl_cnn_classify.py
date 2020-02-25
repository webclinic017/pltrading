import tensorflow as tf
import sys
import os
import pdb
import natsort

window_length = 72
pattern_path = "strict_patterns"
pattern_dir = "/Users/apple/Desktop/dev/projectlife/utils/classification/"+ pattern_path + ".json"
image_path = "/Users/apple/Desktop/dev/projectlife/data/images/candle_tickers/"
image_path_ma = "/Users/apple/Desktop/dev/projectlife/data/images/ma/backtest/"
image_path_renko = "/Users/apple/Desktop/dev/projectlife/data/images/renko/chunks/"
dateformat = '%Y-%m-%d %H:%M'
symbols = ["NANOBTC", "NEOBTC"]#,"WANBTC","ARDRBTC"
interval = "5m"

# def label_one_image():
#     os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#     import tensorflow as tf

#     image_path = sys.argv[1]
#     image_data = tf.gfile.FastGFile(image_path, 'rb').read()
#     label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/output_labels.txt")]
#     with tf.gfile.FastGFile("/tf_files/output_graph.pb", 'rb') as f:
#         graph_def = tf.GraphDef()	## The graph-graph_def is a saved copy of a TensorFlow graph; objektinitialisierung
#         graph_def.ParseFromString(f.read())	#Parse serialized protocol buffer data into variable
#         _ = tf.import_graph_def(graph_def, name='')	# import a serialized TensorFlow GraphDef protocol buffer, extract objects in the GraphDef as tf.Tensor

#     with tf.Session() as sess:

#         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#         predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
#         top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#         for node_id in top_k:
#             human_string = label_lines[node_id]
#             score = predictions[0][node_id]
#             print('%s (score = %.5f)' % (human_string, score))

def label_images():
    label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/output_labels.txt")]
    with tf.gfile.FastGFile("/tf_files/output_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        for symbol in symbols:
            dirFiles = os.listdir(image_path)#backtest
            dirFiles = natsort.natsorted(dirFiles)
            for index,filename in enumerate(dirFiles):
                if filename.endswith(".png") :
                    filename = image_path+filename
                    image_data = tf.gfile.FastGFile(filename, 'rb').read()
                    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
                    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    scores = []
                    for node_id in top_k:
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        if (human_string == "hunter")  and float(score) > 0.80:
                            scores.append([human_string,score])
                    if len(scores) > 0:
                        line = filename + " | " + str(scores)
                        print(line)


# def label_ma():
#     label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/output_labels.txt")]
#     with tf.gfile.FastGFile("/tf_files/output_graph.pb", 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#     with tf.Session() as sess:
#         dirFiles = os.listdir(image_path_ma)
#         dirFiles = natsort.natsorted(dirFiles)
#         balance = 1000
#         #total_balance = 100
#         last_symbol = ""
#         for index,filename in enumerate(dirFiles):
#             if filename.endswith(".png"):
#                 filename = image_path_ma+filename
#                 image_data = tf.gfile.FastGFile(filename, 'rb').read()
#                 softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#                 predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
#                 top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#                 scores = []
#                 symbol = filename.split("BTC")[0].split("/")[-1]
#                 for node_id in top_k:
#                     human_string = label_lines[node_id]
#                     score = predictions[0][node_id]
#                     if (human_string == "found")  and float(score) > 0.99:
#                         scores.append([human_string,score])
#                 if len(scores) > 0:
#                     line = filename + " | " + str(scores)
#                     if ("END" in filename) == False:
#                         # if symbol != last_symbol:
#                         #     total_balance = total_balance + balance
#                         #     balance = 100
#                         profit = float(filename.split("xx")[1].split(".png")[0])
#                         balance = balance + ((balance * profit) / 100)
#                         print(line)
#                         print(balance)
#                         last_symbol = symbol.split('BTC')[0]
#         #print(total_balance)


# def label_renko():
#     label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/output_labels.txt")]
#     with tf.gfile.FastGFile("/tf_files/output_graph.pb", 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#     with tf.Session() as sess:
#         for symbol in symbols:
#             dirFiles = os.listdir(image_path_renko)
#             dirFiles = natsort.natsorted(dirFiles)
#             for index,filename in enumerate(dirFiles):
#                 if filename.endswith(".png") :
#                     image_data = tf.gfile.FastGFile(image_path_renko+filename, 'rb').read()
#                     softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#                     predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
#                     top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
#                     scores = []
#                     for node_id in top_k:
#                         human_string = label_lines[node_id]
#                         score = predictions[0][node_id]
#                         if (human_string == "type1")  and float(score) > 0.50:
#                             scores.append([human_string,score])
#                     if len(scores) > 0:
#                         line = filename + " | " + str(scores)
#                         print(line)

if __name__ == '__main__':
    label_images()
    #label_one_image()
    #label_renko()
