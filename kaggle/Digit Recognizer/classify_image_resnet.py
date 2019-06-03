import tensorflow as tf

import predictor
import input_data

import numpy as np
import pandas as pd

flags = tf.app.flags

flags.DEFINE_string('frozen_inference_graph_path',
                    'F:/Code/buysell/kaggle/Digit Recognizer/model/output_pb/'+
                    'frozen_inference_graph.pb',
                    'Path to frozen inference graph.')

FLAGS = flags.FLAGS


if __name__ == '__main__':
    frozen_inference_graph_path = FLAGS.frozen_inference_graph_path
    
    model = predictor.Predictor(frozen_inference_graph_path)

    df = pd.read_csv('F:/Code/buysell/kaggle/Digit Recognizer/data/test.csv', sep=',')

    all_predictions = []

    reshaped_image = input_data.get_resnet_val(df)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(len(df.index)):
            image = sess.run(reshaped_image)
            print(image.shape)
            pred_label = int(model.predict([image])[0])
            all_predictions.append(pred_label)
        datas = pd.DataFrame(all_predictions)
        datas.to_csv("F:/Code/buysell/kaggle/Digit Recognizer/data/predictions_resnet.csv")
