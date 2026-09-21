import streamlit as st
import sys
import io
from agent import AITutorAutonomousAgent

st.set_page_config(
    page_title="AI Learning Sandbox & Multi-Agent Tutor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Interactive AI Learning Sandbox")
st.caption("Module 5 Multi-Turn Autonomous Tool Loop & Multi-Agent Evaluator")

# Initialize Session State
if "agent" not in st.session_state:
    st.session_state.agent = AITutorAutonomousAgent()
if "lesson" not in st.session_state:
    st.session_state.lesson = None
if "val_result" not in st.session_state:
    st.session_state.val_result = None

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ System Status & Config")
    st.badge("Primary: Gemini 2.5 Flash Lite")
    st.badge("Fallback: OpenRouter Free Models")
    st.badge("Loop: max_steps = 5")
    st.markdown("---")
    
    topic_input = st.text_input("Enter Topic to Learn:", value="Python List Comprehensions")
    generate_btn = st.button("🚀 Start Autonomous Agent Loop", type="primary", use_container_width=True)

# Generate Lesson via Multi-Turn Loop
if generate_btn and topic_input:
    with st.spinner("Autonomous Agent running tool loop (drafting ➔ syntax check ➔ quiz creation)..."):
        st.session_state.lesson = st.session_state.agent.run_autonomous_loop(topic_input, max_steps=5)
        st.session_state.val_result = None

# Render Lesson UI
if st.session_state.lesson:
    lesson = st.session_state.lesson

    # Real-Time Observability Header Metrics
    st.markdown(f"### 📚 Subject: **{lesson['topic']}**")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Autonomous Turns", f"{lesson['turns_taken']} / 5")
    m2.metric("Tools Called", sum(1 for t in lesson['trace'] if 'Tool' in t['action']))
    m3.metric("Status", "Loop Completed" if lesson['turns_taken'] <= 5 else "Max Steps Reached")

    # Step-by-Step Tool Execution Trace Expander
    with st.expander("🔍 Step-by-Step Tool Execution Trace (Observability)", expanded=False):
        for step in lesson["trace"]:
            st.markdown(f"**Turn {step['turn']}** | Provider: `{step['provider']}` | Action: `{step['action']}`")
            if "tool_result" in step:
                st.info(f"Output: {step['tool_result']}")
            st.divider()

    col_left, col_right = st.columns([1, 1], gap="medium")

    # Column 1: Explanation & Quiz Prompt
    with col_left:
        st.subheader("💡 Simple Analogy & Concept")
        st.info(lesson["explanation"])
        
        st.subheader("❓ Self-Check Prompt")
        st.warning(lesson.get("quiz_question", "Explain key concepts in your own words."))

    # Column 2: In-Browser Python Execution Sandbox
# Column 2: In-Browser Executable Python Sandbox with Auto-Installer
    with col_right:
        st.subheader("🐍 Interactive Python Sandbox")
        st.write("Modify the generated code below. Missing libraries will auto-install on execution:")

        raw_code = lesson["code"].replace("```python", "").replace("```", "").strip()
        user_code = st.text_area("Python Editor:", value=raw_code, height=220)

        if st.button("▶️ Execute Code in Sandbox", type="primary"):
            import re
            import subprocess

            def execute_with_autoinstall(code_str: str, max_retries: int = 3) -> str:
                """
                Executes code and automatically installs missing modules dynamically via pip.
                """
                for attempt in range(max_retries):
                    old_stdout = sys.stdout
                    redirected_output = sys.stdout = io.StringIO()
                    
                    try:
                        exec(code_str, {})
                        sys.stdout = old_stdout
                        output = redirected_output.getvalue()
                        return output if output else "[Code executed successfully with no print output]"
                    
                    except ModuleNotFoundError as err:
                        sys.stdout = old_stdout
                        # Extract the missing package name
                        missing_module = err.name
                        
                        # Handle common module mapping aliases (e.g., sklearn -> scikit-learn)
                        module_mappings = {
                            "sklearn": "scikit-learn",
                            "cv2": "opencv-python",
                            "PIL": "pillow",
                            "yaml": "pyyaml",
                            "pywintypes": "pywin32",
                            "win32api": "pywin32",
                            "win32con": "pywin32",
                            "win32gui": "pywin32"
                        }
                        package_to_install = module_mappings.get(missing_module, missing_module)
                        
                        st.info(f"⚙️ Auto-installing missing module: `{package_to_install}`...")
                        
                        # Install package dynamically into active .venv
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", package_to_install
                        ])
                        st.success(f"✅ Successfully installed `{package_to_install}`! Re-running sandbox...")
                    
                    except Exception as e:
                        sys.stdout = old_stdout
                        raise e
                
                return "Failed to resolve missing dependencies after multiple attempts."

            try:
                console_output = execute_with_autoinstall(user_code)
                st.markdown("**Console Output (stdout):**")
                st.code(console_output, language="text")
            except Exception as e:
                st.error(f"Runtime Execution Error: {e}")

    st.markdown("---")

    # Section 3: AI Validation Feedback Box
    st.subheader("📝 Check Your Understanding")
    st.write("Write your explanation of this concept below. The **Validator Agent** will score your output and correct any mistakes.")

    user_summary = st.text_area("Your Explanation:", placeholder="In my own words, this topic works by...", height=100)

    if st.button("Submit Explanation for Review"):
        if not user_summary.strip():
            st.warning("Please enter your explanation before submitting.")
        else:
            with st.spinner("Validator Agent assessing response..."):
                val_data, provider = st.session_state.agent.validate_user_summary(
                    lesson["topic"], user_summary
                )
                st.session_state.val_result = (val_data, provider)

    if st.session_state.val_result:
        val_data, provider = st.session_state.val_result
        st.markdown("### 📊 Assessment Output")
        st.caption(f"Evaluated via `{provider}`")
        
        v_col1, v_col2 = st.columns(2)
        verdict = val_data.get("verdict", "EVALUATED")
        
        if verdict == "CORRECT":
            v_col1.success(f"Verdict: {verdict}")
        elif verdict == "PARTIALLY_CORRECT":
            v_col1.warning(f"Verdict: {verdict}")
        else:
            v_col1.error(f"Verdict: {verdict}")
            
        v_col2.metric("Understanding Score", f"{val_data.get('score', 0)} / 10")
        st.write(val_data.get("feedback", ""))