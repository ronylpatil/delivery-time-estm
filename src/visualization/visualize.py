# implement script to capture performance comparision and dt
# store figures in ./reports/figures


# residual plot
import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def residual_plot(y_test: pandas.Series, y_pred: pandas.Series, path: str) -> None:
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.axhline(y=0, color="black", linestyle="-")
    sns.scatterplot(x=y_pred, y=residuals, color="orange")

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.savefig(path)
