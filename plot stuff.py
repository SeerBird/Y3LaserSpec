from main import plot_it, fit_hyperfine


def plot_magnetic():
    for i in range(5):
        index = i+1
        pre = "data/Session 9/Magnetic variation/"
        post = ".CSV"
        plot_it(f"{pre}GF{index}{post}",
                f"{pre}GP{index}{post}",
                f"{pre}GU{index}{post}")

def plot_protocol_fits():
    windows = [
        [(3.7, 5.2), (7.4, 8.1), (15.4, 15.9), (21.8, 22.7)],
        [(1.9, 3.3), (5.5, 6.2), (13.4, 14.0), (19.9, 20.8)],
        [(1.9, 3.2), (5.5, 6.2), (13.4, 14.0), (19.8, 20.7)],
        [(2.4, 3.8), (6.1, 6.8), (14.0, 14.5), (20.47, 21.18)],
        [(3.3, 4.7), (7.0, 7.7), (14.9, 15.4), (21.2, 22.1)],
        [(2.4, 3.7), (6.0, 6.7), (13.9, 14.5), (20.3, 21.2)],
        [(2.7, 4.0), (6.3, 7.0), (14.2, 14.7), (20.5, 21.5)]
    ]
    for i in range(7):
        index = i+1
        pre = "data/Session 9/"
        post = ".CSV"
        fit_hyperfine(f"{pre}F{index}{post}",
                f"{pre}P{index}{post}",
                f"{pre}U{index}{post}", title = f"Prot{index}", windows = windows[i])
def plot_protocols():
    for i in range(7):
        index = i+1
        pre = "data/Session 9/"
        post = ".CSV"
        plot_it(f"{pre}F{index}{post}",
                f"{pre}P{index}{post}",
                f"{pre}U{index}{post}", title = f"Prot{index}")
plot_protocols()