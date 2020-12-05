# DeepLearningExamples

个人编写的一些深度学习小Demo，欢迎大佬们批评指正。当前有如下Demo，后面会持续更新：

- [使用TensorFlow 2.0 编写的、基于循环神经网络的古诗生成器](https://github.com/AaronJny/DeepLearningSamples/tree/master/tf2-rnn-poetry-generator)
- [使用Keras + BERT 模型编写的古诗生成器](https://github.com/AaronJny/DeepLearningSamples/tree/master/keras-bert-poetry-generator)
- [从零开始编写一个宠物识别系统（爬虫、模型训练和调优、模型部署、Web服务）](https://github.com/AaronJny/pets_classifer)
- [深度学习下的目标检测算法——TensorFlow 2.0下的YOLOv3实践](https://github.com/AaronJny/tf2-keras-yolo3)
- [使用TensorFlow 2.0 编写的图片神经风格迁移](https://github.com/AaronJny/DeepLearningSamples/tree/master/tf2-neural-style-transfer)
- [使用Keras + BERT 模型编写的情感分析Demo](https://github.com/AaronJny/DeepLearningSamples/tree/master/keras-bert-emotional-classifier)
- [使用TensorFlow 2.0 编写的、基于循环神经网络的情感分析Demo](https://github.com/AaronJny/DeepLearningSamples/tree/master/tf2-rnn-emotional-classifier)
- [xyolo:YOLOv3的高度封装版本，几行代码实现模型的训练和调用](https://github.com/AaronJny/xyolo)
- [使用xyolo+孪生神经网络解决点选式验证码](https://github.com/AaronJny/captcha_detection)

这里还有一些非深度学习的有趣的想法：

- [100%成功率的顶象面积验证码识别思路](https://github.com/AaronJny/DeepLearningExamples/tree/master/counter-dingxiang-area-captcha)

# 错误排除

1.当使用PyCharm打开并调试项目时，发现py文件的引用路径有误，怎么办？

答：将当前要调试的小项目的路径设置为`Sources Root`，并取消其他小项目路径的`Sources Root`设置。比如我们要调试的是`keras-bert-poetry-generator`，设置`Sources Root`的方法为`右键点击keras-bert-poetry-generator文件夹 -> Mark Directory as -> Sources Root`，取消的方法同理，取消`Sources Root`的方法为`右键点击keras-bert-poetry-generator文件夹 -> Mark Directory as -> Unmark as Sources Root`。