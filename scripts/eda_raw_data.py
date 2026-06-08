import os
import matplotlib.pyplot as plt

# Paths
input_folder = r"C:\Users\admin\Desktop\3RD YR\GOPI SIR\PROJECT\DATA-2"
output_folder = r"C:\Users\admin\Desktop\3RD YR\GOPI SIR\PROJECT\Plot_2"
os.makedirs(output_folder, exist_ok=True)  # create folder if not exists

# Get all .txt files and sort numerically by number in filename
txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
txt_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # numeric sort

# Loop through sorted txt files
for filename in txt_files:
    file_path = os.path.join(input_folder, filename)

    # Read the file (skip header)
    with open(file_path, "r") as f:
        lines = f.readlines()[1:]

    # Extract X and Y values
    x, y = [], []
    for line in lines:
        vals = line.strip().split()
        if len(vals) == 2:
            try:
                x.append(float(vals[0]))
                y.append(float(vals[1]))
            except ValueError:
                continue

    if not x or not y:
        print(f"⚠️ Skipping {filename} (no valid data)")
        continue

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color="navy", linewidth=0.6, label="Data")
    plt.xlabel("X", fontsize=13)
    plt.ylabel("Y", fontsize=13)
    plt.title(f"Comparison Plot for {filename}", fontsize=15)
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    plt.legend()
    plt.tight_layout()

    # Save as PNG with same base name
    output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Progress printout
    print(f"Saved: {output_file}")

print(f"\nAll plots saved in: {output_folder}")
