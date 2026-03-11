# Fed-batch mAb Production Calculation (Latest)

Inputs
- Titer: 5 g/L
- Working volume: 8000 L
- Train duration: 28 days (4 weeks)
- Product recovery/success: 95%
- Vial size: 200 mg = 0.2 g
- Target output: 100,000 vials per week

Step 1: Protein per train (before recovery)
\[
5\ \text{g/L} \times 8000\ \text{L} = 40{,}000\ \text{g}
\]

Step 2: Protein per train after 95% recovery
\[
40{,}000 \times 0.95 = 38{,}000\ \text{g}
\]

Step 3: Vials per train
\[
38{,}000\ \text{g} \div 0.2\ \text{g/vial} = 190{,}000\ \text{vials/train}
\]

Step 4: Vials per week per train
\[
190{,}000\ \text{vials/train} \div 4\ \text{weeks/train} = 47{,}500\ \text{vials/week/train}
\]

Step 5: Trains needed to reach 100,000 vials/week
\[
100{,}000 \div 47{,}500 = 2.105...
\]

Result
- Minimum whole-number trains required in parallel: **3 trains**

(If runs are sequential across 50 weeks, required number of full train runs would be
\[
\frac{100{,}000\ \text{vials/week} \times 50\ \text{weeks}}{190{,}000\ \text{vials/train}} = 26.3
\]
so **27 train runs** total.)

---

## Earlier Scenarios (same base assumptions unless noted)

### Scenario A: 5 g/L, 8000 L, 28-day train, 70% recovery, 200 mg/vial

1. Product per train (before recovery)
\[
5\ \text{g/L} \times 8000\ \text{L} = 40{,}000\ \text{g}
\]

2. After 70% recovery
\[
40{,}000 \times 0.70 = 28{,}000\ \text{g}
\]

3. Vials per train
\[
28{,}000 \div 0.2 = 140{,}000\ \text{vials/train}
\]

4. 50-week equivalent trains
\[
350\ \text{days} \div 28\ \text{days/train} = 12.5\ \text{trains}
\]

- 12 full trains only: 
  \(12 \times 140{,}000 = 1{,}680{,}000\ \text{vials}\)
- Including partial 13th train (linear): 
  \(12.5 \times 140{,}000 = 1{,}750{,}000\ \text{vials}\)

### Scenario B: 10 g/L, 8000 L, 28-day train, 70% recovery, 200 mg/vial

1. Product per train (before recovery)
\[
10\ \text{g/L} \times 8000\ \text{L} = 80{,}000\ \text{g}
\]

2. After 70% recovery
\[
80{,}000 \times 0.70 = 56{,}000\ \text{g}
\]

3. Vials per train
\[
56{,}000 \div 0.2 = 280{,}000\ \text{vials/train}
\]

4. 50-week equivalent trains
\[
350\ \text{days} \div 28\ \text{days/train} = 12.5\ \text{trains}
\]

- 12 full trains only: 
  \(12 \times 280{,}000 = 3{,}360{,}000\ \text{vials}\)
- Including partial 13th train (linear): 
  \(12.5 \times 280{,}000 = 3{,}500{,}000\ \text{vials}\)
