from seqtree import SequentialTreeSynthesizer


data = [
    {"age": 24, "sex_code": 0, "income_bin": 1, "risk_code": 0},
    {"age": 31, "sex_code": 1, "income_bin": 1, "risk_code": 0},
    {"age": 45, "sex_code": 0, "income_bin": 2, "risk_code": 1},
    {"age": 52, "sex_code": 1, "income_bin": 2, "risk_code": 1},
    {"age": 67, "sex_code": 0, "income_bin": 3, "risk_code": 2},
    {"age": 73, "sex_code": 1, "income_bin": 3, "risk_code": 2},
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
