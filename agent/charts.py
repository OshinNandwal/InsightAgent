import matplotlib.pyplot as plt


def create_bar_chart(series, title, xlabel, ylabel):
    """Create a bar chart from a pandas Series."""

    fig, ax = plt.subplots(figsize=(8, 4.5))

    series.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=0)
    plt.tight_layout()

    return fig


def create_line_chart(series, title, xlabel, ylabel):
    """Create a line chart from a pandas Series."""

    fig, ax = plt.subplots(figsize=(8, 4.5))

    series.plot(
        kind="line",
        marker="o",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig