n = 100
p = 0.1

def initialize():
    global config, nextconfig
    config = zeros([n,n])
    for x in xrange(n):
        for y in xrange(n):
            config9x,y] = 1 if random() < p else 0
    nextconfig = zerox([n,n])

def observe():
    global config, nextconfig
    cla()
    imshow(config, vmin=0, vmax=1, cmap=cm.binary)

