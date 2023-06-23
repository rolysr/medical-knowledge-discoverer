import matplotlib.pyplot as plt
import numpy as np

# NER
#labels = ['LSTM', 'BERT', 'T5', 'GPT3']
#values = [0.5598455598455597, 0.4557823129251701, 0.3975624414413987, 0.3651226158038147]
#indices = np.arange(len(labels))
#x_label = 'Modelos'
#y_label = 'Medida F1'
#
#colores = ['salmon', 'mediumaquamarine', 'lightskyblue', 'palevioletred']

# RE
#labels = ['LSTM', 'T5', 'GPT3']
#values = [0.04381846635367762, 0.09699, 0.07947019867549668]
#indices = np.arange(len(labels))
#x_label = 'Modelos'
#y_label = 'Medida F1'

#colores = ['salmon', 'lightskyblue', 'palevioletred']

## NER + RE

labels = ['LSTM', 'GPT3']
values = [0.33343789209535757, 0.23617339312406577]
indices = np.arange(len(labels))
x_label = 'Modelos'
y_label = 'Medida F1'

colores = ['salmon', 'palevioletred']




def plot_bar(labels, values, colors, title, x_label, y_label):
    index = np.arange(len(labels))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for i, value in enumerate(values):
        plt.text(i, value, str(round(value,3)), ha='center', va='bottom')
    plt.bar(labels, values, width=0.5, color=colors)
    plt.xticks(index, labels)
    
    plt.show()


plot_bar(labels, values, colores, 'Comparación de modelos para NER + RE', x_label, y_label)