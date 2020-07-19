import numpy as np
import sklearn as skl
from scipy import optimize
from tabulate import tabulate
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn.metrics import mean_squared_error
tf.disable_v2_behavior()




def Question1(Output):
    ### QUESTION 1 ###
    
    def func(x):
        return x**6-5*x**4-2*x**3+3*x**2
    
    def Scaled_func(x):
        return (6*x)**6-5*(6*x)**4-2*(6*x)**3+3*(6*x)**2
    
    # Derivative function
    def df(x):
        try:
            value = 6*x**5-20*x**3-6*x**2+6*x
        except OverflowError as e:
            value = OverflowError
        return value
    
    domain = np.linspace(-6,6,num=100000)
    Range = [func(x) for x in domain]
    
    #Find the minimum of the range and the corresponding point in the domain
    Minimum = min(Range)
    IndexOfMin = Range.index(min(Range))
    Output.write("Problem 1 a:\n\tMinimum:\t(" + str(domain[IndexOfMin]) + "," + str(Minimum) + ")\n")
    
    #Part a
    plt.plot(domain, Range)
    plt.plot(domain[IndexOfMin], Minimum, 'ro', label = 'Min = (' + str(domain[IndexOfMin]) + ', ' + str(Minimum) + ' )')
    plt.legend()
    plt.savefig("Problem1A.png")
    plt.show()
    
    #create variables
    next_x = -5.0
    start_x = -5.0
    # We start the search at x=-5
    delta = 0.001  # learning rate
    precision = 0.000001  # Desired precision of result
    max_iters = 1000  # Maximum number of iterations
    next_v= 0.9
    mu= 0.8 #momentum 
    
    #Gradient Descent
    for i in range(max_iters):    
        current_x = next_x
        CurrentDerivative = df(current_x)
        if type(CurrentDerivative) == type(OverflowError):
            finalmin = "nan"
            break
        next_x = current_x - delta * CurrentDerivative
        step = next_x - current_x
        finalmin = next_x
        if abs(step) <= precision:
            break
    Output.write("Problem 1 b:\n\tStart:\t" + str(start_x) + "\n\tdelta:\t" + str(delta) + "\n\tprecision:\t" + str(precision) + "\n\tmaxIters:\t" +str(max_iters) + "\n\tMinimum:\t" + str(finalmin) + "\n")
    
    
    #create variables
    next_x = 5.0
    start_x = 5.0
    # We start the search at x=-5
    delta = 0.001  # learning rate
    precision = 0.000001  # Desired precision of result
    max_iters = 1000  # Maximum number of iterations
    next_v= 0.9
    mu= 0.8 #momentum 
    
    #Gradient Descent
    for i in range(max_iters):    
        current_x = next_x
        CurrentDerivative = df(current_x)
        if type(CurrentDerivative) == type(OverflowError):
            finalmin = "nan"
            break
        next_x = current_x - delta * CurrentDerivative
        step = next_x - current_x
        finalmin = next_x
        if abs(step) <= precision:
            break
    Output.write("Problem 1 c:\n\tStart:\t" + str(start_x) + "\n\tdelta:\t" + str(delta) + "\n\tprecision:\t" + str(precision) + "\n\tmaxIters:\t" +str(max_iters) + "\n\tMinimum:\t" + str(finalmin)+ "\n")
    
    
    #create variables
    next_x = -5.0
    # We start the search at x=-5
    delta = 0.001  # learning rate
    precision = 0.000001  # Desired precision of result
    max_iters = 1000  # Maximum number of iterations
    next_v= 0.9
    start_v = 0.9
    mu= 0.9 #momentum
    
    #NEST
    for i in range(max_iters):    
        current_v = next_v    
        current_x = next_x    
        #next_v = mu*current_v - delta * df(current_x)    
        CurrentDerivative = df(current_x+mu*current_v)
        if type(CurrentDerivative) == type(OverflowError):
            print("Minimum at nan")
            finalmin = "nan"
            break  
        next_v = mu*current_v - delta * CurrentDerivative    
        next_x = current_x + next_v    
        step = next_x - current_x
        finalmin = next_x
        if abs(step) <= precision:        
            break
    Output.write("Problem 1 d:\n\tStart:\t" + str(start_x) + "\n\tdelta:\t" + str(delta)+ "\n\tvelocity:\t" + str(start_v)+ "\n\tmomentum:\t" + str(mu) + "\n\tprecision:\t" + str(precision) + "\n\tmaxIters:\t" +str(max_iters) + "\n\tMinimum:\t" + str(finalmin)+ "\n")
    
    domain = np.linspace(-1,1,num=100000)
    Range = [Scaled_func(x) for x in domain]
    #SANN
    def SA(search_space, func, T):
        scale=np.sqrt(T)
        #start=np.random.choice(search_space)   
        start = np.random.choice(search_space) 
        x=start     
        cur=func(x)    
        history =[x]    
        for i in range(2000):        
            prop=x + np.random.normal()*scale
            if prop > 1 or prop <0 or np.log(np.random.rand())*T >= -(func(prop)-cur):        
            #if prop > 1 or prop <0 or np.random.rand() > np.exp(-(func(prop)-cur)/T):            
                prop = x        
            x=prop        
            cur = func(x)        
            T=0.90*T #reduce Temperature by 1%        
            history.append(x)    
        return x, history
    
    x1, history = SA(domain,Scaled_func, T=10)
    plt.plot(domain, Range)
    plt.plot(domain[IndexOfMin], Minimum, 'ro', label = 'Min = (' + str(domain[IndexOfMin]) + ', ' + str(Minimum) + ' )')
    plt.scatter(x1, Scaled_func(x1), marker='X')
    plt.plot(history, [Scaled_func(x) for x in history])
    plt.title("Scaled function Annealing")
    plt.savefig("SANN.png")
    plt.show()
    print(history[-1])
    Output.write("Problem 1 e:\n\tTemperature:\t10\n\tSchedule:\t.9\n\tMin:\t(" + str(6*x1) + "," + str(func(6*x1)) + ")")
    return



def Question2(optim):

    x_data = np.load('x_hw.npy')
    y_data = np.load('y_hw.npy')
    
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
    optimizer = optim
    train_step = optimizer.minimize(loss)
    
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    #Tensorboard setup
    tf.summary.scalar("LOSS", loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/2/train')
    writer.add_graph(session.graph)
    
    #FASTER: optimizer = N= 200
    N= 10000
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
    MSE = mean_squared_error(AA1*np.sin(ff1*x_data) + AA2*np.sin(ff2*x_data),y_data)
    return MSE




    
Output = open("HW.txt", "w+")
Output.write("Travis Collier, Graduate Student")
Question1(Output)
Output.write('\n\nPROBLEM 2:\n')
optims = [tf.train.GradientDescentOptimizer(0.0001), tf.train.MomentumOptimizer(0.0001, 0.8),tf.train.AdamOptimizer(0.007),tf.train.AdagradOptimizer(0.4),tf.train.AdadeltaOptimizer(0.9),tf.train.RMSPropOptimizer(0.002),tf.train.RMSPropOptimizer(0.002, 0.99),tf.train.RMSPropOptimizer(0.002, 0.99, momentum=0.9)]
results = [(str(optim).replace('tf.train.',''), Question2(optim)) for optim in optims]
Output.write(tabulate(results, headers=["Optimizer", "Loss"]))
