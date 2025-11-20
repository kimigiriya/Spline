import sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout
from PyQt5.uic import loadUi


class PlotWidget(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_spline_comparison(self, x_nodes, y_nodes, x_dense, y_true, spline_data, spline_type):
        self.fig.clear()

        ax = self.fig.add_subplot(111)

        ax.plot(x_dense, y_true, 'k-', label='Исходная функция f(x)', linewidth=2)
        ax.scatter(x_nodes, y_nodes, color='red', s=40, zorder=5, label='Узлы интерполяции')

        colors = {'linear': 'blue', 'parabolic': 'green', 'cubic': 'orange'}
        spline_names = {'linear': 'Линейный сплайн S1(x)',
                        'parabolic': 'Параболический сплайн S2(x)',
                        'cubic': 'Кубический сплайн S3(x)'}

        if spline_data[0] is not None:
            ax.plot(x_dense, spline_data[0], color=colors[spline_type], linewidth=1.5,
                    label=spline_names[spline_type])

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(spline_names[spline_type])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

    def plot_spline_error(self, x_dense, error_data, spline_type):
        self.fig.clear()

        ax = self.fig.add_subplot(111)

        colors = {'linear': 'blue', 'parabolic': 'green', 'cubic': 'orange'}
        spline_names = {'linear': 'Линейный сплайн',
                        'parabolic': 'Параболический сплайн',
                        'cubic': 'Кубический сплайн'}

        if error_data is not None:
            ax.plot(x_dense, error_data, color=colors[spline_type], linewidth=1.5)
            max_error = np.max(error_data)
            ax.set_title(f'{spline_names[spline_type]} - Погрешность\nМакс: {max_error:.2e}')
        else:
            ax.text(0.5, 0.5, 'Данные недоступны', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{spline_names[spline_type]} - Погрешность')

        ax.set_xlabel('x')
        ax.set_ylabel('Погрешность')
        ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

class SplineInterpolator:
    def __init__(self):
        pass

    def linear_spline(self, x_nodes, y_nodes, x_dense):
        n = len(x_nodes)
        y_spline = np.zeros_like(x_dense)

        for i in range(len(x_dense)):
            k = 0
            while k < n - 1 and x_dense[i] > x_nodes[k + 1]:
                k += 1
            x0, x1 = x_nodes[k], x_nodes[k + 1]
            y0, y1 = y_nodes[k], y_nodes[k + 1]
            y_spline[i] = y0 + (y1 - y0) * (x_dense[i] - x0) / (x1 - x0)

        return y_spline

    def parabolic_spline(self, x_nodes, y_nodes, x_dense):
        n = len(x_nodes)
        if n < 3:
            return None

        a, b, c = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n - 1)

        x0, x1, x2 = x_nodes[0], x_nodes[1], x_nodes[2]
        y0, y1, y2 = y_nodes[0], y_nodes[1], y_nodes[2]

        A = ((y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0)) / (
                    (x2 ** 2 - x0 ** 2) * (x1 - x0) - (x1 ** 2 - x0 ** 2) * (x2 - x0))
        B = (y1 - y0 - A * (x1 ** 2 - x0 ** 2)) / (x1 - x0)
        b[0] = 2 * A * x0 + B

        for i in range(n - 1):
            h = x_nodes[i + 1] - x_nodes[i]
            a[i] = y_nodes[i]

            if i < n - 2:
                b[i + 1] = -b[i] + 2 * (y_nodes[i + 1] - y_nodes[i]) / h

            c[i] = (y_nodes[i + 1] - a[i] - b[i] * h) / (h ** 2)

        y_spline = np.zeros_like(x_dense)

        for i in range(len(x_dense)):
            k = 0
            while k < n - 1 and x_dense[i] > x_nodes[k + 1]:
                k += 1
            xk = x_nodes[k]
            y_spline[i] = a[k] + b[k] * (x_dense[i] - xk) + c[k] * (x_dense[i] - xk) ** 2

        return y_spline

    def cubic_spline(self, x_nodes, y_nodes, x_dense):
        n = len(x_nodes)
        if n < 3:
            return None

        A, B, D, F = np.zeros(n - 2), np.zeros(n - 2), np.zeros(n - 2), np.zeros(n - 2)
        a, b, c, d = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n), np.zeros(n - 1)
        alpha, betta, h = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n - 1)

        h[0] = x_nodes[1] - x_nodes[0]
        alpha[0], betta[0] = 0, 0

        for i in range(n - 2):
            h[i + 1] = x_nodes[i + 2] - x_nodes[i + 1]
            A[i] = h[i + 1]
            B[i] = 2 * (h[i] + h[i + 1])
            D[i] = h[i]
            F[i] = 6 * ((y_nodes[i + 2] - y_nodes[i + 1]) / h[i + 1] - (y_nodes[i + 1] - y_nodes[i]) / h[i])

            alpha[i + 1] = -A[i] / (B[i] + alpha[i] * D[i])
            betta[i + 1] = (F[i] - D[i] * betta[i]) / (B[i] + alpha[i] * D[i])

        c[n - 1] = 0
        for i in range(n - 2, -1, -1):
            c[i] = 0 if i == 0 else alpha[i] * c[i + 1] + betta[i]

        for i in range(n - 1):
            a[i] = y_nodes[i + 1]
            d[i] = (c[i + 1] - c[i]) / h[i]
            b[i] = c[i + 1] * h[i] / 2 - (d[i] * h[i] ** 2) / 6 + (y_nodes[i + 1] - y_nodes[i]) / h[i]

        y_spline = np.zeros_like(x_dense)

        for i in range(len(x_dense)):
            k = 0
            while k < n - 1 and x_dense[i] > x_nodes[k + 1]:
                k += 1
            k += 1
            xk = x_nodes[k]
            result = a[k - 1] + b[k - 1] * (x_dense[i] - xk) + (c[k] * (x_dense[i] - xk) ** 2) / 2 + (
                        d[k - 1] * (x_dense[i] - xk) ** 3) / 6
            y_spline[i] = result

        return y_spline

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.spline_interpolator = SplineInterpolator()
        self.current_data = None

        loadUi('desk.ui', self)

        self.init_plots()
        self.connect_signals()

    def init_plots(self):
        self.linear_plot = PlotWidget(self.linearPlotWidget)
        self.parabolic_plot = PlotWidget(self.parabolicPlotWidget)
        self.cubic_plot = PlotWidget(self.cubicPlotWidget)

        linear_layout = QVBoxLayout(self.linearPlotWidget)
        linear_layout.addWidget(self.linear_plot)

        parabolic_layout = QVBoxLayout(self.parabolicPlotWidget)
        parabolic_layout.addWidget(self.parabolic_plot)

        cubic_layout = QVBoxLayout(self.cubicPlotWidget)
        cubic_layout.addWidget(self.cubic_plot)

    def connect_signals(self):
        self.calculateBtn.clicked.connect(self.calculate)
        self.linearToggleBtn.clicked.connect(lambda: self.toggle_plot('linear'))
        self.parabolicToggleBtn.clicked.connect(lambda: self.toggle_plot('parabolic'))
        self.cubicToggleBtn.clicked.connect(lambda: self.toggle_plot('cubic'))

    def toggle_plot(self, spline_type):
        if self.current_data is None:
            return

        if spline_type == 'linear':
            btn = self.linearToggleBtn
            canvas = self.linear_plot
        elif spline_type == 'parabolic':
            btn = self.parabolicToggleBtn
            canvas = self.parabolic_plot
        else:
            btn = self.cubicToggleBtn
            canvas = self.cubic_plot

        current_text = btn.text()
        if current_text == "Показать погрешность":
            btn.setText("Показать сплайн")
            spline_data = self.current_data['spline_results'][{
                'linear': 0, 'parabolic': 1, 'cubic': 2
            }[spline_type]]
            if spline_data[1] is not None:
                canvas.plot_spline_error(
                    self.current_data['x_dense'],
                    spline_data[1],
                    spline_type
                )
        else:
            btn.setText("Показать погрешность")
            self.plot_single_spline(spline_type)

    def plot_single_spline(self, spline_type):
        if self.current_data is None:
            return

        if spline_type == 'linear':
            canvas = self.linear_plot
        elif spline_type == 'parabolic':
            canvas = self.parabolic_plot
        else:
            canvas = self.cubic_plot

        spline_data = self.current_data['spline_results'][{
            'linear': 0, 'parabolic': 1, 'cubic': 2
        }[spline_type]]

        canvas.plot_spline_comparison(
            self.current_data['x_nodes'],
            self.current_data['y_nodes'],
            self.current_data['x_dense'],
            self.current_data['y_true'],
            spline_data,
            spline_type
        )

    def my_eval(self, expr, x_val):
        try:
            safe_dict = {
                'x': x_val,
                'cos': np.cos,
                'sin': np.sin,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'log10': np.log10,
                'sqrt': np.sqrt,
                'pi': np.pi,
                'e': np.e,
                'abs': abs
            }
            return eval(expr, {}, safe_dict)
        except Exception as e:
            raise ValueError(f"Ошибка вычисления функции в точке x={x_val}: {str(e)}")

    def compute_splines(self, x_nodes, y_nodes, x_dense, y_true):
        results = []

        y_linear = self.spline_interpolator.linear_spline(x_nodes, y_nodes, x_dense)
        error_linear = np.abs(y_true - y_linear) if y_linear is not None else None
        max_error_linear = np.max(error_linear) if error_linear is not None else float('inf')
        results.append((y_linear, error_linear, max_error_linear))

        y_parabolic = self.spline_interpolator.parabolic_spline(x_nodes, y_nodes, x_dense)
        error_parabolic = np.abs(y_true - y_parabolic) if y_parabolic is not None else None
        max_error_parabolic = np.max(error_parabolic) if error_parabolic is not None else float('inf')
        results.append((y_parabolic, error_parabolic, max_error_parabolic))

        y_cubic = self.spline_interpolator.cubic_spline(x_nodes, y_nodes, x_dense)
        error_cubic = np.abs(y_true - y_cubic) if y_cubic is not None else None
        max_error_cubic = np.max(error_cubic) if error_cubic is not None else float('inf')
        results.append((y_cubic, error_cubic, max_error_cubic))

        return results

    def calculate(self):
        try:
            a = self.aInput.value()
            b = self.bInput.value()
            n = self.nInput.value()
            func_expr = self.functionInput.text().strip()

            if not func_expr:
                QMessageBox.warning(self, "Ошибка", "Введите функцию f(x)")
                return

            if a >= b:
                QMessageBox.warning(self, "Ошибка", "a должно быть меньше b")
                return

            if n < 3:
                QMessageBox.warning(self, "Ошибка", "Количество узлов должно быть не менее 3")
                return

            x_nodes = np.linspace(a, b, n)
            y_nodes = np.array([self.my_eval(func_expr, x) for x in x_nodes])

            x_dense = np.linspace(a, b, 1000)
            y_true = np.array([self.my_eval(func_expr, x) for x in x_dense])

            spline_results = self.compute_splines(x_nodes, y_nodes, x_dense, y_true)

            self.current_data = {
                'x_nodes': x_nodes,
                'y_nodes': y_nodes,
                'x_dense': x_dense,
                'y_true': y_true,
                'spline_results': spline_results
            }

            for spline_type in ['linear', 'parabolic', 'cubic']:
                self.plot_single_spline(spline_type)
                if spline_type == 'linear':
                    self.linearToggleBtn.setText("Показать погрешность")
                elif spline_type == 'parabolic':
                    self.parabolicToggleBtn.setText("Показать погрешность")
                else:
                    self.cubicToggleBtn.setText("Показать погрешность")

            results_text = f"Функция: f(x) = {func_expr}\n"
            results_text += f"Интервал: [{a}, {b}]\n"
            results_text += f"Количество узлов: n = {n}\n\n"
            results_text += f"МАКСИМАЛЬНЫЕ ПОГРЕШНОСТИ:\n"
            results_text += f"Линейный сплайн S1(x): {spline_results[0][2]:.2e}\n"
            results_text += f"Параболический сплайн S2(x): {spline_results[1][2]:.2e}\n"
            results_text += f"Кубический сплайн S3(x): {spline_results[2][2]:.2e}\n"

            self.resultsText.setText(results_text)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка:\n{str(e)}")

app = QApplication(sys.argv)

try:
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
except Exception as e:
    QMessageBox.critical(None, "Ошибка", f"Не удалось загрузить интерфейс:\n{str(e)}")
    sys.exit(1)