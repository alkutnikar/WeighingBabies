import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import load as l
# dummy data:
#df = pd.DataFrame({'RACEMOM':l.DF_babymod.RACEMOM,'RACEDAD':l.DF_babymod.RACEDAD,'MAGE':l.DF_babymod.MAGE,'FAGE':l.DF_babymod.FAGE,'dv':l.DF_babymod.WGROUP})
df = pd.DataFrame({'RACEMOM':l.DF_babymod.RACEMOM,'WEEKS':l.DF_babymod.WEEKS,'dv':l.DF_babymod.WGROUP})

# create decision tree
dt = DecisionTreeClassifier()
dt =dt.fit(df.ix[:,:2], df.dv)


def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     

     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print node

get_lineage(dt, df.columns)


from sklearn.externals.six import StringIO
with open("dt.dot", 'w') as f:
	f = tree.export_graphviz(dt, out_file=f)


import os
os.unlink('dt.dot')

from sklearn.externals.six import StringIO  
import pydot
from pydot import * 
dot_data = StringIO() 
tree.export_graphviz(dt, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("dt.pdf") 
