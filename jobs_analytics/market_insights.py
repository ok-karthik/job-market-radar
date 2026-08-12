import os
import glob
import json
import argparse
from collections import Counter
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def generate_global_insights(directory_path: str):
    print(f"[*] Scanning directory '{directory_path}' for semantic JSON files...")
    search_pattern = os.path.join(directory_path, "*_semantic.json")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"[-] No files matching '*_semantic.json' found in {directory_path}.")
        return

    print(f"[*] Found {len(files)} semantic JSON files.")
    
    all_jobs = []
    seen = set()
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                jobs = json.load(f)
                for job in jobs:
                    # Global deduplication across all historical files
                    # A job is unique by Title + Company
                    identifier = (job.get('title'), job.get('companyName'))
                    if identifier not in seen:
                        seen.add(identifier)
                        all_jobs.append(job)
            except Exception as e:
                print(f"[-] Error parsing {file}: {e}")

    if not all_jobs:
        print("[-] No jobs found across the files.")
        return

    print(f"[*] Deduplicated into {len(all_jobs)} totally unique jobs across all time.\n")
    
    df = pd.DataFrame(all_jobs)
    
    # 1. Job Counts by Category
    category_counts = df["SemanticCategory"].value_counts()
    
    # 2. Skills Breakdown
    skills_data = {}
    for cat in df["SemanticCategory"].unique():
        cat_df = df[df["SemanticCategory"] == cat]
        cat_total = len(cat_df)
        all_skills = []
        for skills_list in cat_df["TechSkills"]:
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
        
        skill_counts = Counter(all_skills)
        top_skills = skill_counts.most_common(10)
        
        skills_str = ", ".join([f"{skill} ({count/cat_total*100:.0f}%)" for skill, count in top_skills]) if top_skills else "None"
        skills_data[cat] = {"Total Jobs": cat_total, "Top Skills": skills_str}
        
    df_skills = pd.DataFrame.from_dict(skills_data, orient="index")
    if not df_skills.empty:
        df_skills.index.name = "Category"
        # Sort by total jobs descending
        df_skills = df_skills.sort_values(by="Total Jobs", ascending=False)

    # ------------------ PRINTING ------------------
    console.print("[bold green]GLOBAL MARKET INSIGHTS: HISTORICAL DATA[/bold green]")
    
    t = Table(box=box.ROUNDED, show_lines=True)
    t.add_column("Category", justify="left", style="cyan", no_wrap=True)
    t.add_column("Total Unique Jobs", justify="right", style="magenta")
    t.add_column("Top Skills (% of jobs)", justify="left", style="yellow")
    
    if not df_skills.empty:
        for index, row in df_skills.iterrows():
            t.add_row(str(index), str(row["Total Jobs"]), str(row["Top Skills"]))
        console.print(t)
        
    # Save to file
    output_path = os.path.join(directory_path, "global_market_insights.txt")
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("GLOBAL MARKET INSIGHTS: HISTORICAL DATA (DEDUPLICATED)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Unique Jobs Analyzed: {len(all_jobs)}\n")
        f.write("Source Files: " + ", ".join([os.path.basename(f) for f in files]) + "\n\n")
        f.write("Top Skills by Category:\n")
        f.write("-" * 50 + "\n")
        f.write(df_skills.to_string() if not df_skills.empty else "No data")
        
    print(f"\n[+] Global insights saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate aggregated market insights across multiple semantic JSON files.")
    parser.add_argument("directory", nargs="?", default="jobs_output", help="Path to the directory containing *_semantic.json files")
    args = parser.parse_args()
    
    generate_global_insights(args.directory)
