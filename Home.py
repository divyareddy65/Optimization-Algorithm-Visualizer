"""
Home — Optimization Algorithm Visualizer
"""

import streamlit as st

st.set_page_config(page_title="Optimization Visualizer", layout="wide", page_icon="🔍")

st.title("Optimization Algorithm Visualizer - Divya Reddy")
st.markdown("Interactive demonstrations of four optimization algorithms.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1 — Unconstrained Minimization")
    st.markdown("""
    Compare three gradient-based methods on any 2D function:
    - **Steepest Descent (SD)**
    - **Newton's Method**
    - **Conjugate Gradient (CG)**

    Watch their trajectories on a contour plot and compare convergence speed.
    """)
    st.page_link("pages/1_Unconstrained_Minimization.py", label="Open →", icon="📈")

    st.markdown("---")

    st.subheader("3 — Genetic Algorithm")
    st.markdown("""
    Solve the **0/1 Knapsack problem** with a GA:
    - Adjust population size, mutation rate, crossover rate
    - Watch the population evolve over generations
    - See diversity, convergence, and the best packing solution
    """)
    st.page_link("pages/3_Genetic_Algorithm.py", label="Open →", icon="🧬")

with col2:
    st.subheader("2 — Pareto Front")
    st.markdown("""
    Explore **multi-objective optimization** trade-offs:
    - Upload your own CSV or use the built-in smartphone dataset
    - Choose any two objectives and their directions
    - Visualize which solutions are Pareto optimal
    """)
    st.page_link("pages/2_Pareto_Front.py", label="Open →", icon="📊")

    st.markdown("---")

    st.subheader("4 — Simulated Annealing")
    st.markdown("""
    Schedule **10 exams into time slots** with SA:
    - Adjust cooling rate, initial temperature, iterations
    - Watch clashes reduce to zero
    - Compare fast vs slow cooling schedules
    """)
    st.page_link("pages/4_Simulated_Annealing.py", label="Open →", icon="🌡️")
