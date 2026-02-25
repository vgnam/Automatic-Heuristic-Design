import pickle

# Mở file pickle
with open("TSP20.pkl", "rb") as f:
    data_pkl = pickle.load(f)

print(type(data_pkl))  # xem kiểu dữ liệu
print(data_pkl)        # xem nội dung
