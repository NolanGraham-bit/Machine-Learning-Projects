#!/bin/bash
#SBATCH --job-name=kmeans_clustering
#SBATCH --output=kmeans_leap.out  # Output log file
#SBATCH --error=kmeans_leap.err   # Error log file
#SBATCH --time=01:00:00           # Set a maximum runtime (HH:MM:SS)
#SBATCH --partition=compute       # Adjust based on LEAP cluster partition
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --mem=8G                  # Memory allocation
#SBATCH --mail-type=END,FAIL      # Notifications for job completion/failure
#SBATCH --mail-user=your_email@example.com  # Replace with your email

# Load necessary modules (modify as needed)
module load python/3.8  # Adjust based on available versions

# Activate virtual environment (if needed)
# source ~/your_virtual_env/bin/activate

# Run the K-Means clustering script
python kmeans_analysis.py
