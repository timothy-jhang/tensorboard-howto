# tensorboard-howto
tensorflow - how to utilize tensorboard 텐서보드 잘 사용하기

------------------------------------------------------------
# tensorboard 서버 수행하기 - shell 명령으로, logdir를 지정해주어야  함. 기본 서버 포트는 6006인데, 포트를 바꾸려면 --port로 바꿀 수 있음.
% tensorboard --logdir ./cifar10_train --port 8080

------------------------------------------------------------
# tensorboard로 summary들 보기, tensorboard의 클라이언트는 웹브라우저이며, 웹브라우저로 summary를 봄. 
127.0.0.1:6006  포트를 8080으로 바꾸었다면,  127.0.0.1:8080  을 url 주소로 브라우저에 입력함.

-------------------------------------------------------------
# ssh로 접근해서 사용하는 경우에는, desktop화면을 쓸 수 없지만, port forwarding을 다음과 같이 하고, 각자의 컴퓨터 브라우저로 볼 수 있음.
# port forwarding은 shell 명령으로 ssh을 하나 수행해주면 됨. 
% ssh -L 16006:127.0.0.1:6006  sun@server-pc-domain-name-or-ip-address
# 이경우에 각자 피시에서는 
127.0.0.1:16006 을 url 주소로 입력해서 브라우저에서 볼 수 있음.

------------------------------------------------------------------
# tensorflow setting 하기, python code에서 해야 할 일.--->

# 예제 코드는 cifar10_multi_gpu_train.py에서 인용함.
# Build the summary operation from the last tower summaries. 
# dataflow그래프의 말단을 하나 정의함. summary_op, 
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
    summary_op = tf.summary.merge(summaries)
    
# summary writer 열기,     
     summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

# 주기적으로 summary write 동작을 수행함. sess.run을 이용해서. 
    if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
# tf.summary.scalar를 이용해서, scalar 값들을 쓰기 함. 그래프 형태로 global step의 증가에 따른 scalar 값의 변화를 보게됨.
    tf.summary.scalar(name="acc_1off", tensor=acc_1off/FLAGS.batch_size)

# tensor 값들은 직접 보여주기 어려우니까, tf.as_string( )을 사용해서 스트링으로 변환한 후에, tf.summary.text()로 summary에 추가
   tf.summary.text(name="labels",tensor=tf.as_string(labels))
   
# tensorboard 사용하는 기초적인 예제    
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/tensorboard_basic.py
