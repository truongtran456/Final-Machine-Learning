import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error

class RegressionApp:
    # Xử lý giao diện
    def __init__(self, root):
        self.root = root
        self.root.title("Mô hình hồi quy File CSV")
        self.root.geometry("1000x800")

        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(expand=True, fill=tk.BOTH)

        self.controls_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.controls_frame)

        self.result_frame = tk.Frame(self.main_pane)
        self.main_pane.add(self.result_frame)

        self.result_chart = None  
        # Giao diện Button
        self.load_button = tk.Button(self.controls_frame, text="Chọn File CSV", width=15, command=self.load_csv)
        self.load_button.grid(row=0, column=0, padx=(40, 30), pady=10)

        self.execute_button = tk.Button(self.controls_frame, text="Thực hiện", width=15, command=self.execute_regression)
        self.execute_button.grid(row=3, column=0, pady=10)

        self.handle_missing_button = tk.Button(self.controls_frame, text="Xử lý Data Missing Value", width=20, command=self.handle_missing_values)
        self.handle_missing_button.grid(row=4, column=0, pady=10)

        self.vars_frame = tk.Frame(self.controls_frame)
        self.vars_frame.grid(row=1, column=0)

        self.regression_frame = tk.Frame(self.controls_frame)
        self.regression_frame.grid(row=2, column=0)
        
        self.refresh_button = tk.Button(self.result_frame, text="Refresh", command=self.refresh_data)
        self.refresh_button.pack(side=tk.BOTTOM, pady=10)

        # Giao diện hiển thị thông tin Data từ File CSV
        self.result_text = tk.Text(self.result_frame, height=10, width=80)
        self.result_text.pack(expand=False , side=tk.BOTTOM, fill=tk.BOTH, pady=10)
        
        # Giao diện hiển thị thông tin sau khi dùng mô hình hồi quy
        self.result_text2 = tk.Text(self.result_frame, height=5, width=80)
        self.result_text2.pack(expand=True, side=tk.TOP, fill=tk.BOTH, pady=5)
        
        self.data = None
        self.target_var = tk.StringVar()
        self.input_vars = []
        self.regression_types = []

        self.execute_button.grid_remove()

    # Xử lý dữ liệu khi import file CSV vào chương trình
    def load_csv(self):
        # Yêu cầu người dùng chọn file CSV và lưu đường dẫn vào biến file_path
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        # Kiểm tra user đã chọn file CSV
        if not file_path:
            return
        # Đọc dữ liệu từ file CSV và lưu vào biến self.data
        self.data = pd.read_csv(file_path)
        
        if 'Name' in self.data.columns and 'Sex' in self.data.columns:  # Kiểm tra nếu là tập dữ liệu Titanic
            self.preprocess_titanic_data()
            
        # Tạo thông tin về dữ liệu để hiển thị trên giao diện
        info_text = f"Số dòng: {len(self.data)}\nSố cột: {len(self.data.columns)}\n\nKiểu dữ liệu của mỗi cột:\n"
        for column, dtype in self.data.dtypes.items():
            info_text += f"{column}: {dtype}\n"
        
         # Tính số lượng giá trị thiếu trong mỗi cột và thêm vào giao diện
        missing_values = self.data.isnull().sum()
        info_text += "\nSố lượng giá trị thiếu trong mỗi cột:\n"
        for column, missing_count in missing_values.items():
            info_text += f"{column}: {missing_count}\n"
        
        data_preview = "Một số dữ liệu từ file CSV:\n\n"
        data_preview += str(self.data) 
        info_text += "\n\n" + data_preview
        
        # Xóa nội dung hiện tại của result_text2 ( giao diện hiển thị ) và thêm thông tin mới
        self.result_text2.delete('1.0', tk.END)
        self.result_text2.insert(tk.END, info_text)
        
        self.show_variables()
        self.execute_button.grid()

    # Hàm làm mới dữ liệu
    def refresh_data(self):
        if self.data is None:
            messagebox.showerror("Lỗi", "Không có dữ liệu để làm mới.")
            return

        # Hiển thị lại thông tin dữ liệu từ file CSV
        info_text = f"Số dòng: {len(self.data)}\nSố cột: {len(self.data.columns)}\n\nKiểu dữ liệu của mỗi cột:\n"
        for column, dtype in self.data.dtypes.items():
            info_text += f"{column}: {dtype}\n"
        
        missing_values = self.data.isnull().sum()
        info_text += "\nSố lượng giá trị thiếu trong mỗi cột:\n"
        for column, missing_count in missing_values.items():
            info_text += f"{column}: {missing_count}\n"
        
        data_preview = "Một số dữ liệu từ file CSV:\n\n"
        data_preview += str(self.data)
        info_text += "\n\n" + data_preview
        
        self.result_text2.delete('1.0', tk.END)
        self.result_text2.insert(tk.END, info_text)

    def show_variables(self):
        # Xóa tất cả các biến đã có trong chọn biến mục tiêu và độc lập
        for widget in self.vars_frame.winfo_children():
            widget.destroy()
        for widget in self.regression_frame.winfo_children():
            widget.destroy()
        # Kiểm tra xem dữ liệu đã được tải hay chưa
        if self.data is None:
            messagebox.showerror("Lỗi", "Không thể load dữ liệu từ file CSV.")
            return

        # Tạo combobox để chọn biến mục tiêu
        tk.Label(self.vars_frame, text="Chọn biến mục tiêu:").pack()
        target_menu = ttk.Combobox(self.vars_frame, textvariable=self.target_var, values=self.data.columns.tolist(), state='readonly')
        target_menu.pack(expand=True, fill=tk.BOTH)

         # Tạo label và Listbox để chọn biến độc lập
        tk.Label(self.vars_frame, text="Chọn biến độc lập:").pack()
        self.selected_input_vars = []
        self.input_vars_listbox = tk.Listbox(self.vars_frame, selectmode=tk.MULTIPLE)
        for column in self.data.columns:
            self.input_vars_listbox.insert(tk.END, column)
        self.input_vars_listbox.pack()

        # Xử lý nút chọn biến
        select_button = tk.Button(self.vars_frame, text="Chọn biến", command=self.update_selected_inputs)
        select_button.pack()

         # Tạo Listbox để hiển thị các biến độc lập đã chọn
        tk.Label(self.vars_frame, text="Biến độc lập đã chọn:").pack()
        self.selected_vars_listbox = tk.Listbox(self.vars_frame)
        self.selected_vars_listbox.pack()
        
        # Thêm nút Xóa biến đã chọn
        remove_button = tk.Button(self.vars_frame, text="Xóa", command=self.remove_selected_variable)
        remove_button.pack()

        # Tạo Checkbutton để chọn mô hình hồi quy
        tk.Label(self.regression_frame, text="Chọn mô hình hồi quy:").pack()
        self.regression_types = []
        for model_name in ["Hồi quy tuyến tính", "Hồi quy Logistic", "Hồi quy KNN", "Decision Tree"]:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.regression_frame, text=model_name, var=var)
            chk.pack(anchor=tk.W)
            self.regression_types.append((var, model_name))
            
     
    # Hàm xử lý xóa biến độc lập đã chọn
    def remove_selected_variable(self):
        selected_indices = self.selected_vars_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến độc lập để xóa.")
            return

        # Lấy chỉ mục của biến đã chọn
        selected_index = selected_indices[0]

        # Lấy giá trị biến đã chọn
        selected_var = self.selected_vars_listbox.get(selected_index)

        # Xóa biến đã chọn khỏi danh sách biến đã chọn
        self.selected_input_vars.remove(selected_var)

        # Thêm biến đã xóa trở lại vào danh sách ListBox chọn biến độc lập
        self.input_vars_listbox.insert(tk.END, selected_var)

        # Xóa biến đã chọn từ danh sách biến đã chọn trên giao diện
        self.selected_vars_listbox.delete(selected_index)


    # Hàm cập nhật lại biến độc lập có trong danh sách ( nếu chọn biến độc lập thì trên danh sách sẽ cập nhật lại biến )
    def update_selected_inputs(self):
        selected_indices = self.input_vars_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến độc lập.")
            return

        # Lấy chỉ mục của biến đã chọn
        selected_index = selected_indices[0]

        # Lấy giá trị biến đã chọn
        selected_var = self.input_vars_listbox.get(selected_index)

        # Xóa biến đã chọn khỏi danh sách hiển thị
        self.input_vars_listbox.delete(selected_index)

        # Thêm biến đã chọn vào danh sách biến độc lập đã chọn
        self.selected_input_vars.append(selected_var)

        # Cập nhật danh sách biến đã chọn trên giao diện
        self.update_selected_vars_listbox()

    def update_selected_vars_listbox(self):
        # Xóa danh sách biến đã chọn trước đó
        self.selected_vars_listbox.delete(0, tk.END)
        
        # Thêm lại tất cả các biến độc lập đã chọn
        for var in self.selected_input_vars:
            self.selected_vars_listbox.insert(tk.END, var)



    def plot_knn_elbow(self, X, y, max_k=10):
        # Tạo một list để lưu các giá trị MSE
        errors = []
        for i in range(1, max_k):
            knn = KNeighborsRegressor(n_neighbors=i)
            knn.fit(X, y)
            # Dự đoán giá trị của y trên dữ liệu huấn luyện
            pred_i = knn.predict(X)
             # Tính MSE và lưu vào list errors
            errors.append(np.mean((y - pred_i) ** 2))

        # Tạo một biểu đồ để hiển thị Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k), errors, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Elbow Method chọn giá trị "k" tối ưu.')
        plt.xlabel('Số K Neighbors')
        plt.ylabel('Tổng lỗi bình phương')
        plt.show()



    # convert kiểu dữ liệu giới tính
    def preprocess_titanic_data(self):
        # self.data['Title'] = self.data['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
        # title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        # rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Sir', 'Capt', 'Countess', 'Jonkheer', 'Dona']

        # self.data['Title'] = self.data['Title'].map(title_mapping)
        # self.data['Title'].fillna(0, inplace=True) 

        sex_mapping = {'female': 0, 'male': 1}
        self.data['Sex'] = self.data['Sex'].map(sex_mapping)


    def plot_knn_classification(self, X, y, n_neighbors=5):
         # Kiểm tra xem có 2 biến độc lập được chọn không
        if X.shape[1] != 2:
            raise ValueError("Yêu cầu chọn 2 biến độc lập vẽ biểu đồ ranh giới phân loại")
        
        # Đặt bước nhảy cho các chấm biểu thị cho biểu đồ
        h = .02 
        # Tạo màu sắc cho vùng phân loại
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        #Training biểu đồ
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X, y)

         # Xác định giới hạn của biểu đồ
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

         # Dự đoán nhãn cho từng điểm trên mặt phẳng
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)

         # Vẽ các điểm dữ liệu
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"3-Class classification (k = {n_neighbors})")
        plt.show()



    def execute_regression(self):
        # Kiểm tra xem đã load dữ liệu từ file CSV chưa
        if self.data is None:
            messagebox.showerror("Lỗi", "Không thể load dữ liệu từ file CSV.")
            return

         # Lấy tên biến mục tiêu và các mô hình hồi quy được chọn
        target = self.target_var.get()
        selected_models = [model for var, model in self.regression_types if var.get()]

        # Kiểm tra xem có đủ thông tin để thực hiện hồi quy không
        if not target or not selected_models:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến mục tiêu và một mô hình hồi quy.")
            return

         # Lấy danh sách các biến độc lập được chọn
        selected_inputs = self.selected_input_vars  
        if len(selected_inputs) == 0:
            messagebox.showerror("Lỗi", "Vui lòng chọn ít nhất một biến độc lập.")
            return

         # Tạo các mảng numpy cho biến đầu vào và biến mục tiêu
        X = self.data[selected_inputs]
        y = self.data[target]

        # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []
        # Thực hiện hồi quy cho từng mô hình được chọn
        for model_name in selected_models:
            if model_name == "Hồi quy tuyến tính":
                model = LinearRegression().fit(X_train, y_train)
                predictions = model.predict(X_test)
                results.append(f"{model_name} Coef: {model.coef_}, Intercept: {model.intercept_}")
                self.plot_results(X_test, y_test, predictions, f"{model_name} Regression", model=model, target_name=target, input_names=selected_inputs)
            elif model_name == "Hồi quy Logistic":
                model = LogisticRegression().fit(X_train, y_train)
                x_range = np.linspace(X_test.min(), X_test.max(), 300)
                probabilities = model.predict_proba(x_range.reshape(-1, 1))[:, 1]
                plt.plot(x_range, probabilities, label='Xác suất dự đoán')
                plt.scatter(X_test, y_test, color='red', label='Dữ liệu thực tế', alpha=0.5)
                plt.title(f"{model_name} Regression - Đường cong Sigmoid")
                plt.xlabel('X')
                plt.ylabel('Xác suất')
                plt.legend()
                plt.show()
                
                # Vẽ ma trận nhầm lẫn (confusion matrix) và hiển thị báo cáo phân loại
                predictions = model.predict(X_test)
                cm = confusion_matrix(y_test, predictions)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                disp.plot(ax=plt.subplot(), cmap=plt.cm.Blues)
                plt.title(f"Ma trận nhầm lẫn - {model_name}")
                plt.show()
                
                cr = classification_report(y_test, predictions)
                results.append(f"Ma trận nhầm lẫn:\n{cm}\nBáo cáo phân loại:\n{cr}")
            elif model_name == "Hồi quy KNN":
                X_train_selected = X_train[selected_inputs]
                X_test_selected = X_test[selected_inputs]
                if len(selected_inputs) == 1:
                    # Nếu chỉ có 1 biến độc lập, chọn tự chọn giá trị tối ưu cho K
                    mse_for_diff_k = []
                    for k in range(1, 10):
                        model = KNeighborsRegressor(n_neighbors=k)
                        model.fit(X_train_selected, y_train)
                        predictions = model.predict(X_test_selected)
                        mse = mean_squared_error(y_test, predictions)
                        mse_for_diff_k.append(mse)
                    self.plot_knn_elbow(X_train_selected, y_train) 
                    best_k = mse_for_diff_k.index(min(mse_for_diff_k)) + 1
                    model = KNeighborsRegressor(n_neighbors=best_k)
                    model.fit(X_train_selected, y_train)
                    predictions = model.predict(X_test_selected)
                    mse = mean_squared_error(y_test, predictions)
                    results.append(f"{model_name} - Best K: {best_k} - Predictions: {predictions[:5]} - MSE: {mse}") 
                    self.plot_results(X_test_selected, y_test, predictions, f"{model_name} Regression", model=model, target_name=self.target_var.get(), input_names=selected_inputs)

                 # Nếu có 2 biến độc lập, vẽ biểu đồ phân loại và biểu đồ elbow
                elif len(selected_inputs) == 2:
                    self.plot_knn_classification(X_train_selected.to_numpy(), y_train.to_numpy(), n_neighbors=5)
                    self.plot_knn_elbow(X_train_selected, y_train) 
                # xử lý phần decision tree
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                cm = confusion_matrix(y_test, predictions)
                results.append(f"Decision Tree Confusion Matrix:\n{cm}")
                plt.figure(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=selected_inputs, class_names=np.unique(y).astype(str), rounded=True, fontsize=12)
                plt.show()
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, "\n".join(results))


    def plot_results(self, X, y, y_pred, title, model=None, target_name=None, input_names=None):
        fig, ax = plt.subplots()

        xlabel = input_names[0] if input_names else 'Biến độc lập'
        ylabel = target_name if target_name else 'Biến mục tiêu'

         # Trường hợp hồi quy tuyến tính xử lý 1 hoặc 2 biến độc lập
        if model and isinstance(model, LinearRegression):
            # Nếu chỉ có một biến độc lập, vẽ biểu đồ scatter plot và đường hồi quy
            if len(input_names) == 1:
                ax.scatter(X.iloc[:, 0], y, color='blue', label='Actual')
                ax.plot(X.iloc[:, 0], y_pred, color='red', label='Predicted')
                ax.set_xlabel(xlabel) 
            elif len(input_names) == 2:
                # Nếu có hai biến độc lập, vẽ biểu đồ scatter plot 3D
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(X[input_names[0]], X[input_names[1]], y, color='blue', label='Actual')
                ax.scatter(X[input_names[0]], X[input_names[1]], y_pred, color='red', label='Predicted')
                ax.set_xlabel(input_names[0])
                ax.set_ylabel(input_names[1])
                ax.set_zlabel(ylabel)
                ax.set_title(title)
                ax.legend()
                plt.show()
            else:
                ax.plot(X, y_pred, color='red', label='Predicted')
                ax.set_xlabel('Biến độc lập')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            plt.show()
            
        # Trường hợp hồi quy Logistic với 1 biến độc lập
        elif model and hasattr(model, "predict_proba") and X.shape[1] == 1:
            x_values = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 300)
            y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
            ax.plot(x_values, y_values, color='green', label='Sigmoid')
            ax.set_xlabel(input_names[0]) 
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            plt.show()
        elif isinstance(model, KNeighborsRegressor):
            if len(input_names) == 1:
                # Nếu chỉ có một biến độc lập, vẽ biểu đồ scatter plot và đường thẳng nối điểm dữ liệu
                ax.scatter(X.iloc[:, 0], y, color='blue', label='Actual')
                ax.scatter(X.iloc[:, 0], y_pred, color='red', label='Predicted')
                for i in range(len(X)):
                    x_i = X.iloc[i, 0] if isinstance(X, pd.DataFrame) else X[i]
                    y_i = y.iloc[i] if isinstance(y, pd.Series) else y[i]
                    y_pred_i = y_pred[i]
                    ax.plot([x_i, x_i], [y_i, y_pred_i], color='red')
                ax.set_xlabel(input_names[0]) 
                 # Nếu có nhiều hơn một biến độc lập, vẽ biểu đồ scatter plot và đường thẳng nối điểm dữ liệu
            else:
                ax.scatter(X, y, color='blue', label='Actual')
                ax.scatter(X, y_pred, color='red', label='Predicted')
                for i in range(len(X)):
                    ax.plot([X.iloc[i, j] for j in range(X.shape[1])], [y.iloc[i], y_pred[i]], color='red')
                ax.set_xlabel('Biến độc lập')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            plt.show()
        else:
            ax.scatter(X, y, color='blue', label='Actual')
            ax.plot(X, y_pred, color='red', label='Predicted')
            if len(input_names) == 1:
                ax.set_xlabel(input_names[0])
            else:
                ax.set_xlabel('Biến độc lập')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            plt.show()

    def handle_missing_values(self):
        # Kiểm tra nếu không có dữ liệu được tải từ file CSV
        if self.data is None:
            messagebox.showerror("Lỗi", "Không thể xử lý giá trị thiếu vì chưa load dữ liệu từ file CSV.")
            return

        # Tạo một cửa sổ mới khi bấm xử lý data missing value
        missing_values_window = tk.Toplevel(self.root)
        missing_values_window.title("Xử lý giá trị thiếu")
        missing_values_window.geometry("600x400")

        # Tạo danh sách cột có giá trị thiếu
        columns_with_missing_values = self.data.columns[self.data.isnull().any()].tolist()

        # Biến lưu trữ cột được chọn từ combobox
        selected_column_var = tk.StringVar()
        tk.Label(missing_values_window, text="Chọn cột có giá trị thiếu:").pack(pady=(0, 5))
        column_menu = ttk.Combobox(missing_values_window, textvariable=selected_column_var, values=columns_with_missing_values, state='readonly')
        column_menu.pack(pady=(0, 20))

        # Biến lưu trữ giá trị được nhập vào entry
        value_entry_var = tk.StringVar()
        tk.Label(missing_values_window, text="Hoặc nhập giá trị cụ thể:").pack(pady=(0, 5))
        value_entry = tk.Entry(missing_values_window, textvariable=value_entry_var)
        value_entry.pack(pady=(0, 20))

        # Hàm áp dụng giá trị vào giá trị thiếu
        def apply_value(specific_value=None):
            
            selected_column = selected_column_var.get()
            if specific_value is None:
                specific_value = value_entry_var.get()
            if selected_column and specific_value:
                  # Điền giá trị cụ thể vào giá trị thiếu trong cột đã chọn
                self.data[selected_column].fillna(specific_value, inplace=True)
                 # Hiển thị thông báo khi điền giá trị thành công 
                messagebox.showinfo("Thông báo", f"Giá trị Missing Value trong cột '{selected_column}' đã được điền bằng '{specific_value}'.")
                missing_values_window.destroy()
                 # Hiển thị lại danh sách biến và refresh data
                self.show_variables()
                self.refresh_data()

        # Hàm xóa dòng chứa giá trị thiếu
        def delete_row():
            selected_column = selected_column_var.get()
            if selected_column:
                 # Xóa các dòng chứa giá trị thiếu trong cột đã chọn
                self.data = self.data[self.data[selected_column].notnull()]
                messagebox.showinfo("Thông báo", f"Các dòng chứa giá trị Missing Value trong cột '{selected_column}' đã được xóa.")
                missing_values_window.destroy()
                self.show_variables()
                self.refresh_data()

        apply_button = tk.Button(missing_values_window, text="Áp dụng", width=15, command=apply_value)
        apply_button.pack(pady=(0, 20))

        tk.Label(missing_values_window, text="Chọn các giá trị để điền vào Missing Value").pack(pady=(0, 20))
        button_frame = tk.Frame(missing_values_window)
        button_frame.pack(side=tk.TOP, pady=(0, 10))

         # Button xử lý giao diện và điền giá trị trung bình, trung vị, mode và button xóa dòng
         
        mean_button = tk.Button(button_frame, text="Mean", width=15, command=lambda: apply_value(self.data[selected_column_var.get()].mean()))
        mean_button.pack(side=tk.LEFT, padx=15)

        median_button = tk.Button(button_frame, text="Median", width=15, command=lambda: apply_value(self.data[selected_column_var.get()].median()))
        median_button.pack(side=tk.LEFT, padx=15)

        mode_button = tk.Button(button_frame, text="Mode", width=15, command=lambda: apply_value(self.data[selected_column_var.get()].mode().iloc[0]))
        mode_button.pack(side=tk.LEFT, padx=15)

        delete_row_button = tk.Button(missing_values_window, text="Xóa dòng", width=15, command=delete_row)
        delete_row_button.pack(pady=(10, 0))

         # Chạy vòng lặp chính của cửa sổ xử lý giá trị thiếu
        missing_values_window.mainloop()



if __name__ == "__main__":
    root = tk.Tk()
    app = RegressionApp(root)
    root.mainloop()
