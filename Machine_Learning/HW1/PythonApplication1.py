
import tensorflow as tf
from numpy import load
from numpy import sin as npsin,pi
import matplotlib.pyplot as plt
from tensorboard import program

x_data = load('x_hw.npy')
y_data = load('y_hw.npy')

HW = open('HW.txt', 'w+')
HW.write('MACHINE LEARNING PS1\nTRAVIS COLLIER\n25 SEP 2019\n\nQUESTION 1: ATTACHED\n\n')

#Uncomment to plot the raw data Answer to question 1
#plt.plot(x_data,y_data)
#plt.show()

#Initialize the waves to separate values or else they will train symmetrically
A1 = tf.Variable(tf.ones([1]))
f1 = tf.Variable(tf.ones([1]))
A2 = tf.Variable(tf.zeros([1]))
f2 = tf.Variable(tf.ones([1]))

#This could be a more efficent method for the hypothesis, but it's so small who cares.
#FASTER: y = add(multiply(A1,sin(multiply(f1,x_data))), multiply(A2,sin(multiply(f2,x_data))))
#Answer to Question 2
y = A1*tf.sin(f1*x_data) + A2*tf.sin(f2*x_data)
loss = tf.reduce_mean(tf.square(y - y_data))

#Could realistically get away with a training step as large as .2 or higher, 
#but I'll do this to show I understand the danger of large training steps.

#FASTER: optimizer = tf.train.AdamOptimizer(0.2)
optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Tensorboard setup
tf.summary.scalar("LOSS", loss)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/2/train')
writer.add_graph(session.graph)

#FASTER: optimizer = N= 200
N= 1500
for k in range(N):
    session.run(train_step)
    s=session.run(merged)
    writer.add_summary(s, k)
    if k%200 == 0:
        print("k =",k, "A1 =", session.run(A1[0]), "A2 =", session.run(A2[0]),"f1 =", session.run(f1[0]), "f2 =", session.run(f2[0]), "loss =",session.run(loss) )

AA1,ff1,AA2,ff2 = session.run([A1,f1,A2,f2])
AA1 = AA1[0]
AA2 = AA2[0]
ff1 = ff1[0]
ff2 = ff2[0]

HW.write("QUESTION 2: The wave equation is: y = " + str(AA1) + "sin(" + str(ff1) + "x) + " + str(AA2) + "sin(" + str(ff2) + "x)\n\n")

#Answer to question 3
plt.plot(x_data,y_data, alpha=0.2)
plt.plot(x_data,AA1*npsin(ff1*x_data) + AA2*npsin(ff2*x_data))
plt.legend(['Raw_data', 'Result'])
plt.show()

HW.write("QUESTION 3: ATTACHED\n\n")

#Answer to Question 4
HW.write("QUESTION 4 : Value of regression at x=0.6pi: " + str(AA1*npsin(ff1*.6*pi) + AA2*npsin(ff2*.6*pi)) + '\n\n')

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', "./logs/2/train"])
url = tb.launch()

#Answer to Question 5
HW.write("QUESTION 5 : One obviously sees that we could have gotten away with 600 iterations\n\n")
HW.write("REMARK: Another interesting observation is that we could have gotten away with a much larger training step since the loss function is highly convergent.\n\n")
HW.close()
