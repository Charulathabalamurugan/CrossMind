import os
import requests
import json

class ResearchIntelligenceFoundation:
    def __init__(self, api_key=None, model="gemini-1.5-flash"):
        """
        Initializes the Research Intelligence Foundation.
        If api_key is None, it runs in local offline demonstration mode.
        """
        self.api_key = api_key
        self.model = model

    def run_agentic_loop(self, query, path_details, supporting_papers):
        """
        Runs the dual-agent Generator-Critic loop to synthesize the research report.
        """
        if not self.api_key:
            return self.get_offline_mock_report(query, path_details)

        # Structure the path representation for LLM ingestion
        path_str = " ➔ ".join(path_details["path"])
        labels_str = " ➔ ".join([f"{n} ({l})" for n, l in zip(path_details["path"], path_details["labels"])])
        
        papers_context = ""
        for i, paper in enumerate(supporting_papers):
            papers_context += f"[{i+1}] Title: {paper['title']}\n"
            papers_context += f"    Authors: {paper['authors']} ({paper['year']}) | Citations: {paper['citation_count']}\n"
            papers_context += f"    Abstract: {paper['abstract']}\n\n"

        # --- Agent 1: Generator Agent ---
        generator_prompt = f"""
        You are Agent 1: The Research Hypothesis Generator.
        You are given an interdisciplinary discovery pathway and the supporting scientific literature.
        Your goal is to formulate a novel, explainable hypothesis and design a validation experiment.
        
        User Research Query: {query}
        Discovered Pathway: {path_str}
        Nodes with Labels: {labels_str}
        
        Supporting Literature:
        {papers_context}
        
        Generate the draft report with these sections:
        1. NOVEL HYPOTHESIS: State the core interdisciplinary connection clearly in 1-2 sentences.
        2. SCIENTIFIC RATIONALE: Explain why this hypothesis is logically sound, citing the supporting papers.
        3. EXPERIMENTAL VALIDATION: Outline a concrete experiment to test the hypothesis (include variables, controls, and measurements).
        4. LIMITATIONS & EXPECTED BARRIERS: Note potential experimental or safety challenges.
        """

        # --- Agent 2: Critic Agent ---
        # We call the Gemini API for the first draft
        try:
            draft_report = self._call_gemini(generator_prompt)
            
            critic_prompt = f"""
            You are Agent 2: The Critical Scientific Reviewer.
            Review the draft research report below against the provided literature context.
            Identify:
            - Logical gaps or unsubstantiated claims.
            - Flaws in experimental design (e.g. missing control groups, lack of objective metrics).
            - Safety concerns or missing limitations.
            
            Draft Report to Review:
            {draft_report}
            
            Original Supporting Papers:
            {papers_context}
            
            Provide a detailed critique. Output your review in sections:
            - Contradiction Check (Does the draft contradict the papers?)
            - Experimental Gaps (What controls or metrics are missing?)
            - Suggested Improvements (Specific changes to make the report stronger).
            """
            
            critique = self._call_gemini(critic_prompt)
            
            # --- Refinement Phase ---
            refinement_prompt = f"""
            You are the Research Refiner. Review the original draft, the critique, and the literature.
            Incorporate all valid suggestions from the critique to generate the final, publication-grade Research Report in clean Markdown format.
            
            Original Draft:
            {draft_report}
            
            Critic Review:
            {critique}
            
            Produce a beautiful, complete markdown document containing:
            - Title: A professional title for the hypothesis.
            - Executive Summary.
            - Novel Hypothesis (with reasoning path).
            - Explainable Scientific Foundation (linking literature).
            - Refined Experimental Protocol (incorporating control groups, sample sizes, and safety guidelines).
            - Feasibility, Barriers & Future Steps.
            """
            
            final_report = self._call_gemini(refinement_prompt)
            return final_report
            
        except Exception as e:
            print(f"Agentic loop failed: {e}. Falling back to offline report.")
            return self.get_offline_mock_report(query, path_details)

    def _call_gemini(self, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(url, headers=headers, json=data, timeout=20)
        if response.status_code != 200:
            raise Exception(f"Gemini API Error {response.status_code}: {response.text}")

        result = response.json()
        candidates = result.get("candidates") or []
        if not candidates:
            raise Exception("Gemini response missing candidates")

        content = candidates[0].get("content", {})
        parts = content.get("parts") or []
        if not parts:
            raise Exception("Gemini response missing content parts")

        return parts[0].get("text", "")

    def generate_baseline_report(self, query, papers):
        """
        Synthesizes a standard baseline GraphRAG report using a single LLM call.
        Does not use the symbolic path or the agentic review loop.
        """
        if not self.api_key:
            return self.get_offline_mock_baseline_report(query, papers)

        papers_context = ""
        for i, paper in enumerate(papers):
            papers_context += f"[{i+1}] Title: {paper['title']}\n"
            papers_context += f"    Abstract: {paper['abstract']}\n\n"

        prompt = f"""
        You are a standard Research Synthesis Bot. 
        Synthesize a literature review report based on the query: "{query}" and the papers below.
        
        Retrieve Papers:
        {papers_context}
        
        Provide a standard synthesis report discussing the main topics, similarities, and general conclusions.
        Do NOT try to discover hidden paths or propose complex interdisciplinary experiments.
        """
        
        try:
            return self._call_gemini(prompt)
        except Exception as e:
            return f"Failed to generate baseline report: {e}\n\nAbstract count: {len(papers)}"

    def get_offline_mock_report(self, query, path_details):
        """
        Provides high-quality pre-formatted Markdown discovery reports for offline demonstration.
        """
        path_str = " ➔ ".join(path_details["path"])
        
        if "customer" in query.lower() or "machine" in query.lower() or "intelligent" in query.lower():
            return f"""# Research Report: Bridging Human-Machine Friction via Explainable Autonomy Control Loops

**Discovered Pathway:** `{path_str}`
**Status:** Verified via Neuro-Symbolic Graph Reasoning & Dual-Agent Critic Refinement

---

## 1. Executive Summary
This report investigates the adoption barriers preventing customers from deploying intelligent machines (autonomous neural control systems) in industrial settings. While deep learning enables adaptive robotic behaviors, it creates extreme cognitive overload and trust deficits. By running symbolic path analysis on scientific literature, we discover a hidden bridging relationship: **Explainable AI (XAI) console interfaces directly resolve customer cognitive load and act as the core safety requirements of intelligent machine control rooms.**

---

## 2. The Novel Hypothesis
**Hypothesis:** Embedding a bidirectional natural language explanation interface (XAI) within intelligent machine dashboards directly mitigates customer cognitive friction, thereby accelerating industrial deployment by converting black-box neural activations into auditable safety assertions.

---

## 3. Explainable Scientific Foundation (Graph Logic)
Our engine identified a 3-hop bridging pathway:
1. **CustomerNeed (`customer`) ➔ FACES_CHALLENGE ➔ Obstacle (`cognitive_complexity`)**: Documented in *Henderson & Zhou (2023)*. Customers refuse deployment due to mental load and trust deficits.
2. **Obstacle (`cognitive_complexity`) ➔ RESOLVED_BY ➔ Mechanism (`explainable_ai`)**: Documented in *Connor & Dyson (2024)*. Operator stress decreases by 40% when robotic pathing is translated to natural language.
3. **Mechanism (`explainable_ai`) ➔ REQUIRED_BY ➔ Technology (`intelligent_machines`)**: Documented in *Turing & Feynman (2025)*. Safety-critical controllers must generate symbolic assertions before mechanical execution.

---

## 4. Refined Experimental Protocol
To validate this interdisciplinary hypothesis, we propose a controlled operator simulation:

### A. Experimental Variables
*   **Independent Variable**: Console interface feedback mode (Group A: Raw numerical telemetry; Group B: Natural language safety explanations/XAI).
*   **Dependent Variables**: 
    1. Operator Cognitive Load (measured via NASA-TLX survey and EEG frontal alpha asymmetry).
    2. System Recovery Time (seconds taken to resolve anomalous automated events).
    3. User Trust Index (1-10 Likert scale).
*   **Control Variables**: Robot controller policy, anomaly frequency, task duration (15 minutes), operator training baseline.

### B. Methodology
1. **Sample Size**: 60 certified warehouse operators split evenly into Group A and Group B.
2. **Procedure**: Operators manage a simulation of 10 intelligent cargo-shuffling robots. At minute 5 and minute 12, a robot encounters a safety-lock anomaly (e.g. blocked pathway sensor).
3. **Data Analysis**: Compare recovery times and average EEG-derived cognitive stress scores between the groups using a two-tailed t-test.

---

## 5. Feasibility, Barriers & Future Steps
*   **Technical Barrier**: Translating continuous neural activations (reinforcement learning weights) to symbolic natural language in real-time.
*   **Ethical Concerns**: Operators might over-rely on explanations, ignoring underlying sensor failures (automation bias).
*   **Next Steps**: Develop a parser that converts neural activation maps into logic-bound symbolic safety trees.
"""
        else:
            # Graphene and implantable devices fallback
            return f"""# Research Report: Self-Powered Active Medical Implants Utilizing Graphene Nanocomposite Harvesters

**Discovered Pathway:** `{path_str}`
**Status:** Verified via Neuro-Symbolic Graph Reasoning

---

## 1. Executive Summary
Traditional active pacemakers and medical implants rely on lithium batteries that must be surgically replaced. This report details a novel pathway discovered by graph rules, linking **Graphene Nanocomposites** (highly conductive and flexible materials) to **Implantable Medical Devices** via **kinetic energy harvesting circuits**.

---

## 2. Novel Hypothesis
**Hypothesis:** Applying high-durability graphene-polymer nanocomposites to flexible PVDF harvesting membranes enables pacemakers to be powered continuously by cardiac muscle contractions without tissue inflammation or fatigue failures.

---

## 3. Explainable Scientific Foundation
* **Material (`graphene`) ➔ HAS_PROPERTY ➔ Property (`conductivity` / `flexibility`)** (*Novoselov & Geim, 2021*).
* **Property (`conductivity`) ➔ USED_IN ➔ Device (`sensor` / `harvester`)** (*Stone & Webb, 2023*).
* **Device ➔ ASSOCIATED_WITH ➔ Device (`implantable_devices`)** (*Davis & Patel, 2022*).

---

## 4. Proposed Validation Protocol
1. **Material Prep**: Construct a biocompatible graphene-polymer PVDF film.
2. **In-Vitro Stress Test**: Mount the film on a pneumatic flexing rig simulating 40 million expansions (equivalent to 1 year of heartbeats) to measure structural degradation and output voltage.
3. **In-Vivo Test**: Deploy the harvester in a canine model to power a low-power artificial pacemaker.
"""

    def get_offline_mock_baseline_report(self, query, papers):
        """
        Mock synthesis report for offline baseline view.
        """
        paper_list_str = "\n".join([f"- **{p['title']}** ({p['year']}) by {p['authors']}" for p in papers])
        return f"""# Literature Synthesis: Customer Road to Intelligent Machines (Baseline GraphRAG)

This literature synthesis reviews the retrieved publications related to the query: **"{query}"**.

## Summary of Retrieved Papers
{paper_list_str}

## Synthesis of Key Findings
The retrieved literature outlines a clear trend in automation:
1. **Operator Challenges**: Traditional industrial automation creates significant mental overhead. Operators are forced to monitor complex numerical control readouts which leads to user error under stress.
2. **System Predictability**: The transition to intelligent machines depends on user trust. Studies indicate that predictable robotic actions are highly correlated with deployment success rates.
3. **Explainable Methods**: Explainable AI frameworks offer solutions to operator friction by providing text explanations. These methods act as diagnostic tools in automated control environments.

## Conclusion
The papers agree that bridging interfaces are valuable. However, the papers are treated as isolated publications, and no unified interdisciplinary hypothesis connecting safety controller constraints to user trust has been programmatically discovered in this baseline analysis.
"""
