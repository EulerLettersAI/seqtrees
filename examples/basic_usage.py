from seqtree import SequentialTreeSynthesizer


data = [
    {"age_band": "young", "sex": "F", "risk": "low"},
    {"age_band": "young", "sex": "M", "risk": "low"},
    {"age_band": "middle", "sex": "F", "risk": "medium"},
    {"age_band": "middle", "sex": "M", "risk": "medium"},
    {"age_band": "older", "sex": "F", "risk": "high"},
    {"age_band": "older", "sex": "M", "risk": "high"},
]


model = SequentialTreeSynthesizer(
    optimize_order=True,
    tree_backend="auto",
    n_jobs=-1,
    random_state=42,
    min_samples_leaf=1,
)
model.fit(data)

print("Order:", model.get_variable_order())
print("Synthetic rows:")
for row in model.sample(5):
    print(row)
