from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('C', 'I'), ('I', 'A'), ('C', 'A')])

# Defining individual CPDs.
cpd_r = TabularCPD(variable='C', variable_card=2, values=[[0.995], [0.005]])
cpd_u = TabularCPD(variable='I', variable_card=2, values=[[0.99, 0.03], [0.01, 0.97]], evidence = ['C'], evidence_card=[2])

# The CPD for C is defined using the conditional probabilities based on U and R
cpd_c = TabularCPD(variable='A', variable_card=2,
                   values=[[0.001, 0.5, 0.98, 0.3],
                           [0.999, 0.95, 0.2, 0.97]],
                  evidence=['C', 'I'],
                  evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_r, cpd_u, cpd_c)

print("CPD for Cutremur:")
print(cpd_r)

print("CPD for Incendiu:")
print(cpd_u)

print("CPD for AlarmaIncendiu:")
print(cpd_c)

# Print the model structure
print("Model structure:")
print(model.edges())


# Verifying the model
assert model.check_model()



infer = VariableElimination(model)
result = infer.query(variables=['C'], evidence={'A': 1})
print(result)


infer = VariableElimination(model)
result = infer.query(variables=['I'], evidence={'A': 0})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

