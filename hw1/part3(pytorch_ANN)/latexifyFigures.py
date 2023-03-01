import os

with open("figures.txt", "w+") as f:
    for filename in sorted(os.listdir()):
        if filename[-4:] == ".png":
            config = filename[:-4].split("_")

            accloss = config[-1]
            others = config[:-1]

            captionStart = f"Loss values" if accloss == "loss" else "Accuracy values"
            caption = f"{captionStart} for {', '.join(others)}"
            print(
                '''\\begin{figure}[H]
    \centering
    \includegraphics[width=\\textwidth]{pictures/%s}
    \caption{%s}
    \label{fig:%s}
\end{figure}''' % (filename, caption, filename[:-4]), file=f)
