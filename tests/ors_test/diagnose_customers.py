#!/usr/bin/env python3
"""
Interactive ORS Diagnostic (Bisect Mode) - FIXED PATHS
------------------------------------------------------
1. Loads geocoded data from 'tests/ors_test/'.
2. Attempts to fetch the Distance Matrix for the whole group.
3. If successful -> Checks for unreachable ('None') customers.
4. If FAILED -> Pauses and asks to split the group (Bisect).

Usage:
    python tests/ors_test/diagnose_customers.py
"""

import sys
import os
import pandas as pd
import logging
import glob
import time

# --- FIX: Setup Imports from Project Root ---
# Get the directory where this script lives (.../tests/ors_test)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up TWO levels to reach the project root (.../wearhouse-optimizer)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Add the root to Python's search path so we can find 'config' and 'calculations'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from calculations.ors import ORSHandler

# Configure logging (Mute internal logs)
logging.basicConfig(level=logging.CRITICAL) 

# Global list to collect failures
problematic_customers = []

def get_yes_no(prompt):
    """Helper to get user confirmation."""
    while True:
        try:
            choice = input(f"{prompt} [y/n]: ").lower().strip()
            if choice in ['y', 'yes']:
                return True
            if choice in ['n', 'no']:
                return False
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

def analyze_success(matrix, candidates):
    """Check a successful matrix response for 'None' values (Unreachable)."""
    local_problems = 0
    distances = matrix.get('distances', [])
    
    if not distances:
        return

    # Check connectivity of each candidate
    for i in range(len(candidates)):
        dist = distances[0][i]
        c = candidates[i]
        
        if dist is None:
            print(f"💀 [UNREACHABLE] ID: {c['id']} | {c['name']}")
            problematic_customers.append({
                'id': c['id'],
                'name': c['name'],
                'issue': 'Unreachable (None)',
                'details': f"Loc: {c['lat']},{c['lng']}"
            })
            local_problems += 1
            continue

        if dist > 1000000: # 1000km
             print(f"⚠️ [EXTREME DIST] ID: {c['id']} | {dist/1000:.0f}km")
             problematic_customers.append({
                'id': c['id'],
                'name': c['name'],
                'issue': 'Extreme Distance',
                'details': f"{dist/1000:.0f}km"
            })
             local_problems += 1

    if local_problems == 0:
        print(f"   ✅ Chunk of {len(candidates)} is healthy.")

def diagnose_recursive(ors, candidates, depth=0):
    """
    Recursive function to test a batch of customers.
    If it fails, asks user to split.
    """
    indent = "  " * depth
    count = len(candidates)
    
    if count == 0:
        return

    print(f"\n{indent}🔄 Testing batch of {count} customers...")
    
    # Prepare coordinates for ORS
    coords = [(c['lat'], c['lng']) for c in candidates]

    try:
        # ATTEMPT REQUEST
        # We use a modest timeout (10s). If the server hangs on size, it will trigger an exception.
        matrix = ors.get_distance_matrix(coords, timeout=10)
        
        # IF SUCCESS: Analyze contents
        analyze_success(matrix, candidates)
        return

    except Exception as e:
        print(f"{indent}❌ Batch FAILED (Size: {count})")
        print(f"{indent}   Error: {str(e)[:100]}...") 
        
        # Base case: If 1 item failed, we know exactly who it is
        if count == 1:
            c = candidates[0]
            print(f"{indent}   👉 CULPRIT FOUND: {c['id']} - {c['name']}")
            problematic_customers.append({
                'id': c['id'],
                'name': c['name'],
                'issue': 'API Crash/Failure',
                'details': str(e)
            })
            return

        # Recursive step: Ask user
        print(f"{indent}   ⚠️  The system caused an error.")
        if get_yes_no(f"{indent}   Split this batch into two groups of {count//2}?"):
            mid = count // 2
            left = candidates[:mid]
            right = candidates[mid:]
            
            diagnose_recursive(ors, left, depth + 1)
            diagnose_recursive(ors, right, depth + 1)
        else:
            print(f"{indent}   ⏭️  Skipping this batch.")

def run_simulation():
    print(f"\n🚀 Starting ORS Interactive Diagnostic")
    print("=" * 70)

    # 1. Locate Files in the SAME directory as the script
    # tests/ors_test/*.csv
    csv_files = glob.glob(os.path.join(script_dir, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in: {script_dir}")
        return

    target_file = max(csv_files, key=os.path.getsize)
    print(f"📂 Loaded: {os.path.basename(target_file)}")

    # 2. Init ORS
    api_key = Config.OPENROUTESERVICE_API_KEY
    ors = ORSHandler(api_key)

    # 3. Load & Prep Data
    try:
        df = pd.read_csv(target_file)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"❌ Load error: {e}")
        return

    # Convert DF to list of dicts for easier splitting
    all_candidates = []
    for idx, row in df.iterrows():
        try:
            lat = float(row.get('lat'))
            lng = float(row.get('lng'))
            # Filter valid coords only
            if pd.isna(lat) or pd.isna(lng) or lat == 0 or lng == 0:
                continue 
            
            all_candidates.append({
                'id': row.get("מס' לקוח", idx),
                'name': row.get("שם לקוח", "Unknown"),
                'lat': lat,
                'lng': lng
            })
        except:
            continue
            
    print(f"   Valid Geocoded Rows: {len(all_candidates)}")
    print("-" * 70)
    print("ℹ️  Instructions:")
    print("   - I will try to send ALL customers to ORS.")
    print("   - If it works, great.")
    print("   - If it crashes, I will ask you to split the group.")
    print("-" * 70)

    # 4. Start Recursion
    diagnose_recursive(ors, all_candidates)

    # 5. Final Report
    print("\n" + "=" * 70)
    print(f"📊 SUMMARY")
    print(f"   Problematic Customers Found: {len(problematic_customers)}")
    
    if problematic_customers:
        out_csv = os.path.join(script_dir, "problematic_customers.csv")
        pd.DataFrame(problematic_customers).to_csv(out_csv, index=False)
        print(f"📝 Saved to: {out_csv}")

if __name__ == "__main__":
    run_simulation()