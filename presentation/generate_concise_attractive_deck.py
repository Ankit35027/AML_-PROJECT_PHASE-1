"""
Generates an attractive, professional 9-slide PowerPoint presentation
for Historical METR-LA Traffic Speed Forecasting (Phase 1 Baselines & Phase 2 ST-GAT).
Each member presents 2 to 3 slides, with Slide 3 as the dedicated Phase 1 summary.
All speaker notes contain conversational Hinglish scripts for viva defense.
"""

from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "reports/figures"
IMP_DOCS_DIR = REPO_ROOT / "imp_DOCS"

# Modern Attractive Palette
NAVY_HEADER = RGBColor(15, 23, 42)       # #0F172A (Deep Slate)
CARD_BG = RGBColor(248, 250, 252)        # #F8FAFC (Clean light surface)
CARD_BORDER = RGBColor(203, 213, 225)    # #CBD5E1 (Subtle border)
TEXT_DARK = RGBColor(15, 23, 42)         # #0F172A
TEXT_MUTED = RGBColor(100, 116, 139)     # #64748B
TEXT_LIGHT = RGBColor(248, 250, 252)     # #F8FAFC
BLUE_ACCENT = RGBColor(37, 99, 235)      # #2563EB (Electric Blue)
GREEN_ACCENT = RGBColor(16, 185, 129)    # #10B981 (Emerald Green for ST-GAT wins)
AMBER_ACCENT = RGBColor(245, 158, 11)    # #F59E0B (Guardrail/Notice)

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, presenter_tag, category="METR-LA TRAFFIC SPEED FORECASTING • 4-MEMBER RESEARCH TEAM"):
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = NAVY_HEADER
        top_bar.line.fill.background()

        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(8.5), Inches(0.3))
        p_cat = tb_cat.text_frame.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(8.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = RGBColor(148, 163, 184)

        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(8.5), Inches(0.65))
        p_title = tb_title.text_frame.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_LIGHT

        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.6), Inches(0.32), Inches(2.9), Inches(0.52))
        badge.fill.solid()
        badge.fill.fore_color.rgb = BLUE_ACCENT
        badge.line.fill.background()
        p_badge = badge.text_frame.paragraphs[0]
        p_badge.text = presenter_tag
        p_badge.font.size = Pt(10.5)
        p_badge.font.bold = True
        p_badge.font.color.rgb = TEXT_LIGHT
        p_badge.alignment = PP_ALIGN.CENTER

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.2)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: TITLE & EXECUTIVE OVERVIEW (Member 1: Ankit Garg)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_header(s1, "Historical METR-LA Traffic Speed Forecasting", "MEMBER 1: ANKIT GARG", "COMPREHENSIVE TWO-PHASE SCIENTIFIC STUDY")

    add_card(s1, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
    tb1_left = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf1_left = tb1_left.text_frame
    tf1_left.word_wrap = True

    p = tf1_left.paragraphs[0]
    p.text = "Project Scope & Four-Member Attribution"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(12)

    team_items = [
        ("Member 1 (Ankit Garg - ankit35027):", "Lead: Data Provenance, Missingness Protocol & Lineage Manifest (WS1)"),
        ("Member 2 (Yashpal - Yashpal-2004):", "Lead: Causal Feature Windowing & Spatial GAT Layer Scaffolding (WS2)"),
        ("Member 3 (Anshika - anshika292005):", "Lead: Temporal Sequence Modeling (GRU) & Multi-Horizon Forecast Heads (WS3)"),
        ("Member 4 (Samriddhi Shah - samriddhi-techy):", "Lead: Deep Training Harness, Masked Loss, Dashboard & Systems Integration (WS4)")
    ]
    for role, desc in team_items:
        p = tf1_left.add_paragraph()
        p.text = f"• {role}\n  {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(10)

    add_card(s1, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
    tb1_right = s1.shapes.add_textbox(Inches(7.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf1_right = tb1_right.text_frame
    tf1_right.word_wrap = True

    p = tf1_right.paragraphs[0]
    p.text = "Academic Boundary & Scientific Guardrails"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = AMBER_ACCENT
    p.space_after = Pt(12)

    scope_items = [
        ("Strict Empirical Grounding:", "The benchmark Caltrans METR-LA dataset records strictly freeway traffic speed in miles per hour (mph). It does NOT record traffic signal phases, cycle timings, or queue lengths."),
        ("Zero Ungrounded Signal Claims:", "In strict compliance with academic honesty, our research focuses exclusively on multi-horizon traffic speed prediction (+15m, +30m, +60m lookaheads). No false claims of municipal signal control are asserted."),
        ("Two-Phase Rigorous Methodology:", "• Phase 1: Zero-leakage classical machine learning baselines (Persistence, Ridge, XGBoost) and spatial neighbor ablation.\n• Phase 2: Spatio-Temporal Graph Attention Networks (ST-GAT) learning dynamic edge weights over road topology."),
        ("Reproducibility:", "Evaluated on 90,149 held-out test instances with 100% deterministic reproducibility.")
    ]
    for h, desc in scope_items:
        p = tf1_right.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(10)

    s1.notes_slide.notes_text_frame.text = (
        "[MEMBER 1 - ANKIT GARG (SLIDE 1 - INTRO)]\n\n"
        "\"Good morning respected professors and committee members. Mera naam Ankit Garg hai, aur main apne group members Yashpal, "
        "Anshika, aur Samriddhi Shah ke saath hamara project present kar raha hoon: 'Historical METR-LA Traffic Speed Forecasting'.\n\n"
        "Shuruat mein hi main ek bohot important academic guardrail clarify karna chahta hoon: Hamara course title traffic signal control "
        "mention karta hai, lekin Caltrans METR-LA benchmark data mein strictly inductive loop detectors ki speed (miles per hour) record hoti hai. "
        "Isme municipal signals ya queue lengths nahi hain. Isliye scientific truthfulness maintain karte hue, hamara research strictly "
        "multi-horizon freeway traffic speed prediction par based hai.\n\n"
        "Humne is project ko do rigorous phases mein execute kiya hai: Phase 1 mein verified Classical ML baselines, aur Phase 2 mein "
        "Spatio-Temporal Graph Attention Networks (ST-GAT). Sabhi 4 members ne dedicated workstreams deliver kiye hain jo hamari repository par committed hain.\""
    )

    # =========================================================================
    # SLIDE 2: DATASET & MISSINGNESS AUDIT (Member 1: Ankit Garg)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "METR-LA Dataset Lineage & Missingness Accounting", "MEMBER 1: ANKIT GARG")

    stat_boxes = [
        ("207 Loop Detectors", "Highway Sensor Network", BLUE_ACCENT),
        ("34,272 Timestamps", "Continuous 5-Min Intervals (4 Months)", NAVY_HEADER),
        ("8.11% Zero-Speed Records", "Sensor Dropouts Audited (0 NaNs)", AMBER_ACCENT)
    ]
    for i, (val, lbl, color) in enumerate(stat_boxes):
        x = Inches(0.8 + i * 3.95)
        add_card(s2, x, Inches(1.4), Inches(3.8), Inches(1.1), bg_color=CARD_BG)
        tb_stat = s2.shapes.add_textbox(x + Inches(0.2), Inches(1.45), Inches(3.4), Inches(0.9))
        tf_s = tb_stat.text_frame
        p_val = tf_s.paragraphs[0]
        p_val.text = val
        p_val.font.size = Pt(17)
        p_val.font.bold = True
        p_val.font.color.rgb = color
        p_lbl = tf_s.add_paragraph()
        p_lbl.text = lbl
        p_lbl.font.size = Pt(9.5)
        p_lbl.font.color.rgb = TEXT_MUTED

    add_card(s2, Inches(0.8), Inches(2.7), Inches(5.7), Inches(4.3))
    tb2_left = s2.shapes.add_textbox(Inches(1.0), Inches(2.8), Inches(5.3), Inches(4.0))
    tf2_left = tb2_left.text_frame
    tf2_left.word_wrap = True

    p = tf2_left.paragraphs[0]
    p.text = "Cryptographic Provenance & Zero Protocol"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    audit_points = [
        ("Source & Period:", "Caltrans Performance Measurement System (PeMS) District 7 (Los Angeles freeways: I-5, I-10, I-405, US-101) from March 1 to June 27, 2012."),
        ("Cryptographic Lineage:", "Raw HDF5 speed matrix verified with SHA-256 hash (64784b76...) and manifest logged in data_lineage.json."),
        ("Physical Missingness Protocol:", "Dataset contains 0 NaNs and 0 impossible speeds (>100 mph). However, 575,302 readings (8.11%) are exactly 0.0 mph."),
        ("Label vs Feature Handling:", "• Historical lags use causal forward-fill imputation to avoid missing input features.\n• Ground-truth targets strictly require speed > 0.0 mph to ensure fair test evaluation without dropout distortion.")
    ]
    for h, desc in audit_points:
        p = tf2_left.add_paragraph()
        p.text = f"• {h} {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s2, Inches(6.8), Inches(2.7), Inches(5.7), Inches(4.3))
    tb2_right = s2.shapes.add_textbox(Inches(7.0), Inches(2.8), Inches(5.3), Inches(4.0))
    tf2_right = tb2_right.text_frame
    tf2_right.word_wrap = True

    p = tf2_right.paragraphs[0]
    p.text = "Chronological 70/10/20 Split Isolation"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    split_points = [
        ("Zero Future Leakage Principle:", "Traffic data is inherently non-stationary. Random k-fold cross-validation leaks future information into past predictions. We enforce strictly chronological temporal splits:"),
        ("Training Set (70%):", "March 1, 2012 to May 23, 2012 (23,990 timestamps; 359,505 pilot samples). All feature standardizers are fitted exclusively on this partition."),
        ("Validation Set (10%):", "May 23, 2012 to June 4, 2012 (3,427 timestamps; 51,060 samples). Dedicated strictly to hyperparameter tuning and early stopping."),
        ("Held-Out Test Set (20%):", "June 4, 2012 to June 27, 2012 (6,855 timestamps; 90,149 valid target observations). Frozen and evaluated only once."),
        ("Passing Unit Tests:", "Validated by tests/test_splits_and_leakage.py passing 6/6 scientific verification tests.")
    ]
    for h, desc in split_points:
        p = tf2_right.add_paragraph()
        p.text = f"• {h} {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s2.notes_slide.notes_text_frame.text = (
        "[MEMBER 1 - ANKIT GARG (SLIDE 2 - DATA & AUDIT)]\n\n"
        "\"Ab aate hain hamare dataset aur data integrity protocol par. METR-LA benchmark mein Los Angeles ke 207 highway inductive "
        "loop detectors ka data hai, jisme March se June 2012 tak har 5 minute ka traffic speed recorded hai—total 34,272 timestamps.\n\n"
        "Humne data ka thorough missingness audit kiya: Dataset mein 0 NaNs the, lekin 8.11% readings exactly 0.0 mph thi. "
        "Real-world highway par 0.0 mph ka matlab ya toh sensor breakdown hota hai ya road closure. Isliye humne causal forward-fill "
        "use kiya feature lags ke liye, aur ground-truth evaluation mein sirf valid speeds (>0.0 mph) ko rakha.\n\n"
        "Sabse crucial part: Humne strict chronological 70% Train, 10% Validation, aur 20% Test partitioning enforce ki hai. "
        "StandardScaler sirf training split par fit kiya gaya taaki future data ka 0% leakage ho. Ye 6 automated unit tests se validated hai. "
        "Ab main Yashpal ko hand over karta hoon Phase 1 baselines aur GAT architecture explain karne ke liye.\""
    )

    # =========================================================================
    # SLIDE 3: PHASE 1 SUMMARY (Member 2: Yashpal) -- DEDICATED 1-PAGE SUMMARY
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Phase 1 Summary: Classical ML & Spatial Discovery", "MEMBER 2: YASHPAL", "DEDICATED PHASE 1 EMPIRICAL MILESTONE")

    add_card(s3, Inches(0.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb3_1 = s3.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf3_1 = tb3_1.text_frame
    tf3_1.word_wrap = True

    p = tf3_1.paragraphs[0]
    p.text = "1. Causal Feature Pipeline"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    p1_items = [
        ("Historical Speed Lags:", "Causal sliding window extracting 12 past steps (past 60 minutes: t-11 to t) for forecast origin t."),
        ("Cyclical Time Features:", "Sin/Cos transforms for Time-of-Day (diurnal patterns) and Day-of-Week (weekday vs weekend)."),
        ("Multi-Horizon Lookaheads:", "+15 min (offset +3), +30 min (offset +6), and +60 min (offset +12)."),
        ("Classical Architectures:", "• Persistence: y_hat(t+h) = y(t)\n• Ridge Regression: L2 penalty tuned on val set\n• XGBoost: 100 gradient boosted trees with early stopping")
    ]
    for h, desc in p1_items:
        p = tf3_1.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s3, Inches(4.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb3_2 = s3.shapes.add_textbox(Inches(5.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf3_2 = tb3_2.text_frame
    tf3_2.word_wrap = True

    p = tf3_2.paragraphs[0]
    p.text = "2. Multi-Horizon Benchmarks"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    p2_items = [
        ("15-Minute Lookahead (+3):", "• Persistence: 3.0011 mph\n• Ridge: 2.8789 mph\n• XGBoost: 2.5909 mph (13.7% error drop vs persistence)"),
        ("30-Minute Lookahead (+6):", "• Persistence: 3.6157 mph\n• Ridge: 3.6050 mph\n• XGBoost: 3.1668 mph (12.4% error drop)"),
        ("60-Minute Lookahead (+12):", "• Persistence: 4.7302 mph\n• Ridge: 4.9066 mph (fails due to collinearity)\n• XGBoost: 4.0083 mph (15.3% error drop)"),
        ("Peak vs Off-Peak Surge:", "During rush hours, errors surge by 50-65% across all models (XGBoost 60m error rises to 5.09 mph).")
    ]
    for h, desc in p2_items:
        p = tf3_2.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s3, Inches(8.8), Inches(1.4), Inches(3.7), Inches(5.6), bg_color=CARD_BG, border_color=GREEN_ACCENT)
    tb3_3 = s3.shapes.add_textbox(Inches(9.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf3_3 = tb3_3.text_frame
    tf3_3.word_wrap = True

    p = tf3_3.paragraphs[0]
    p.text = "3. Spatial Ablation Discovery"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(8)

    p3_items = [
        ("The Spatial Experiment:", "We compared an isolated sensor model against an augmented model taking lags from top-3 road network topological neighbors (from Caltrans distance graph)."),
        ("Empirical Findings:", "• 15m MAE: 2.59 -> 2.55 mph\n• 30m MAE: 3.17 -> 3.07 mph\n• 60m MAE: 4.02 -> 3.81 mph (-5.2% relative error reduction!)"),
        ("Physical Significance:", "Proves that highway traffic congestion propagates as downstream shockwaves."),
        ("Direct Bridge to Phase 2:", "Static neighbor concatenation helped, but traffic shockwaves travel dynamically. This created the direct motivation for Graph Attention Networks!")
    ]
    for h, desc in p3_items:
        p = tf3_3.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s3.notes_slide.notes_text_frame.text = (
        "[MEMBER 2 - YASHPAL (SLIDE 3 - PHASE 1 SUMMARY)]\n\n"
        "\"Thank you Ankit. Respected examiners, main Slide 3 mein Phase 1 ki complete empirical summary present kar raha hoon.\n\n"
        "Phase 1 mein hamara objective tha verified classical ML baselines establish karna. Humne past 12 lags (past 60 minutes) aur "
        "cyclical time features ko causal window se extract kiya, aur teen models benchmark kiye: Persistence, Ridge, aur XGBoost. "
        "Test set ke 90,149 instances par XGBoost ne best accuracy di: 15m par 2.59 mph aur 60m par 4.01 mph. Yahan Ridge collinearity "
        "ki wajah se 60m par Persistence se bhi kharab perform kar raha tha (4.91 mph).\n\n"
        "Phase 1 ka sabse bada scientific breakthrough tha hamara 'Spatial Feature Ablation': Jab humne sensor ke sath uske road network "
        "ke top-3 neighbors ke lags inject kiye, toh 60-minute MAE 4.02 se ghat kar 3.81 mph ho gaya—pura 5.2% error reduction! "
        "Isne scientifically prove kiya ki traffic jams highway par shockwaves ki tarah aage badhte hain. Aur yahi discovery hamara "
        "direct bridge bani Phase 2 ke Graph Attention Networks ke liye.\""
    )

    # =========================================================================
    # SLIDE 4: PHASE 2 DYNAMIC GAT FORMULATION (Member 2: Yashpal)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Phase 2 Formulation: Moving Beyond Static Tabular ML", "MEMBER 2: YASHPAL")

    add_card(s4, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
    tb4_left = s4.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf4_left = tb4_left.text_frame
    tf4_left.word_wrap = True

    p = tf4_left.paragraphs[0]
    p.text = "Why Classical ML Hits a Hard Ceiling"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    ceil_items = [
        ("1. Static vs. Dynamic Shockwaves:", "Classical tabular models assign fixed regression weights to neighbors. But in real life, traffic shockwaves travel at variable speeds (10-25 mph upstream) depending on real-time traffic density."),
        ("2. Time-Invariant Spatial Receptive Field:", "A sensor 2 miles upstream is critically important during a severe morning bottleneck, but completely irrelevant during free-flow midnight conditions. Tabular models cannot dynamically turn attention on and off."),
        ("3. Flattening Destroys Graph Topology:", "Concatenating neighbor columns into a 2D matrix flattens the physical highway graph, ignoring directed road connectivity, ramp splits, and spatial distances."),
        ("The Solution: Dynamic Graph Attention:", "Instead of fixed weights, compute dynamic edge attention coefficients alpha_ij(t) at every single timestamp based on real-time traffic conditions!")
    ]
    for h, desc in ceil_items:
        p = tf4_left.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    add_card(s4, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
    tb4_right = s4.shapes.add_textbox(Inches(7.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf4_right = tb4_right.text_frame
    tf4_right.word_wrap = True

    p = tf4_right.paragraphs[0]
    p.text = "Dynamic Spatial Attention Formulation"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    gat_math_items = [
        ("1. Attention Energy Computation:", "e_ij^(t) = LeakyReLU( a^T [ W h_i^(t) || W h_j^(t) ] )\nWhere W projects node features, a is a learnable attention vector, and || represents feature concatenation."),
        ("2. Adjacency-Masked Softmax Normalization:", "alpha_ij^(t) = exp( e_ij^(t) ) / sum_{k in N_i} exp( e_ik^(t) )\nAttention weights are normalized strictly over physical road network neighbors N_i. Non-connected sensors are masked with -infinity."),
        ("3. Multi-Head Aggregation (K=4 Heads):", "h_i' = ||_{k=1}^K sigma( sum_{j in N_i} alpha_ij^(k) W^(k) h_j )\nFour independent attention heads capture different traffic dynamics (incident shockwaves vs commute flow)."),
        ("4. Guaranteed Conservation Law:", "sum_{j in N_i} alpha_ij^(t) = 1.0, verified by automated unit tests in test_phase2_gat.py.")
    ]
    for h, desc in gat_math_items:
        p = tf4_right.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    s4.notes_slide.notes_text_frame.text = (
        "[MEMBER 2 - YASHPAL (SLIDE 4 - GAT FORMULATION)]\n\n"
        "\"Ab sawal yeh uthta hai: Agar XGBoost 2.59 mph achieve kar raha tha, toh hume Graph Attention Network ki kya zaroorat padi?\n\n"
        "Classical machine learning ek hard ceiling par hit karta hai kyunki tabular models static hote hain. Ek upstream sensor subah ke "
        "rush hour jam mein bohot critical hota hai, lekin raat ko 2 baje bilkul irrelevant. Classical model feature weights ko time ke saath "
        "change nahi kar sakta. Iske alawa tabular feature concatenation highway ke physical road network topology ko destroy kar deta hai.\n\n"
        "Isliye Phase 2 mein humne Multi-Head Graph Attention Network formulate kiya. Isme attention energy e_ij node features ke interaction "
        "se compute hoti hai, aur road adjacency matrix se mask karke row-softmax calculate kiya jata hai taaki attention weights alpha_ij "
        "har sensor ke neighbors par strictly 1.0 sum karein. Humne K=4 independent attention heads implement kiye hain. "
        "Ab Anshika aapko end-to-end architecture aur heatmaps explain karengi.\""
    )

    # =========================================================================
    # SLIDE 5: ST-GAT PIPELINE & TRAINING (Member 3: Anshika)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "ST-GAT End-to-End Pipeline & Training Harness", "MEMBER 3: ANSHIKA")

    add_card(s5, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
    tb5_left = s5.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf5_left = tb5_left.text_frame
    tf5_left.word_wrap = True

    p = tf5_left.paragraphs[0]
    p.text = "SpatioTemporalGAT Architecture (70,403 Params)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    arch_items = [
        ("1. Temporal Sequence Backbone (GRU):", "Input batch: (B, T=12, N=207, F=5). A causal Gated Recurrent Unit (hidden_dim=64) processes past 12 time steps, extracting temporal context for each highway sensor."),
        ("2. Multi-Head Dynamic Spatial GAT:", "4-Head Graph Attention Layer operating over sensor adjacency tensor A in R^{N x N}. Dynamically computes alpha_ij(t) to route spatial shockwaves across freeway corridors."),
        ("3. Multi-Horizon Direct Linear Heads:", "Direct projection heads output speed predictions simultaneously for +15m, +30m, and +60m in physical mph units without autoregressive error compounding."),
        ("4. Lightweight & Deployable:", "Total 70,403 trainable parameters—fast inference, low memory footprint, and fully reproducible checkpoint serialized to models/st_gat_best_checkpoint.pt.")
    ]
    for h, desc in arch_items:
        p = tf5_left.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    add_card(s5, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.6))
    tb5_right = s5.shapes.add_textbox(Inches(7.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf5_right = tb5_right.text_frame
    tf5_right.word_wrap = True

    p = tf5_right.paragraphs[0]
    p.text = "Optimization Suite & Masked Loss Harness"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    opt_items = [
        ("Masked MAE Objective Function:", "L_masked = sum( |y - y_hat| * M ) / ( sum(M) + eps )\nGradient backpropagation occurs STRICTLY over valid sensor observations (M == True, speed > 0.0 mph). Missing loop detector dropouts cannot destabilize weight updates."),
        ("Optimizer & Learning Rate Schedule:", "• AdamW optimizer (learning rate 1e-3, weight decay 1e-4)\n• ReduceLROnPlateau scheduler (halving LR on validation stagnation)\n• Gradient norm clipping (max_norm = 5.0) to prevent exploding gradients during bottleneck transitions"),
        ("Validation Early Stopping:", "Monitors validation MAE with early stopping patience to prevent overfitting to training splits."),
        ("Passing Scientific Tests:", "Validated by tests/test_phase2_gat.py verifying tensor shapes, attention weight normalization, and loss masking.")
    ]
    for h, desc in opt_items:
        p = tf5_right.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    s5.notes_slide.notes_text_frame.text = (
        "[MEMBER 3 - ANSHIKA (SLIDE 5 - ARCHITECTURE & TRAINING)]\n\n"
        "\"Thank you Yashpal. Respected committee members, main Anshika hoon aur main SpatioTemporalGAT architecture aur hamara "
        "training harness present kar rahi hoon.\n\n"
        "Hamara deep learning model teen cohesive components se milkar bana hai: Pehla, ek Causal GRU temporal backbone jisme 64 hidden "
        "units hain, jo har sensor ke past 12 time steps (60 minutes) ki speed aur cyclical time features ko encode karta hai. "
        "Dusra, hamara 4-Head Spatial Graph Attention layer, jo freeway adjacency graph par dynamic attention weights calculate karta hai. "
        "Aur teesra, direct multi-horizon heads jo +15m, +30m, aur +60m ki speeds simultaneous output karte hain bina autoregressive error compounding ke. "
        "Pura model sirf 70,403 parameters ka hai, jo lightweight aur highly efficient hai.\n\n"
        "Training ke liye humne custom Masked MAE loss function implement kiya. Iska fayda yeh hai ki 8.11% zero-speed readings jo sensor dropouts thi, "
        "unpar gradient update zero ho jata hai aur model sirf valid physical traffic par learn karta hai. AdamW optimizer aur ReduceLROnPlateau "
        "scheduler ke sath model converge hua aur best checkpoint models/st_gat_best_checkpoint.pt par saved hai.\""
    )

    # =========================================================================
    # SLIDE 6: DYNAMIC ATTENTION HEATMAPS (Member 3: Anshika) -- EMBEDDED FIGURE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Dynamic Spatial Attention Heatmaps: Tracking Shockwaves", "MEMBER 3: ANSHIKA")

    add_card(s6, Inches(0.8), Inches(1.4), Inches(5.2), Inches(5.6))
    tb6 = s6.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(4.8), Inches(5.3))
    tf6 = tb6.text_frame
    tf6.word_wrap = True

    p = tf6.paragraphs[0]
    p.text = "Physical Shockwave Interpretability"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    heat_items = [
        ("Dynamic Attention Interpretability:", "Graph Attention Networks provide full transparency: we can inspect the exact attention weights alpha_ij(t) assigned by any sensor to its neighbors at any timestamp."),
        ("Midnight Free-Flow (02:00 AM - Right):", "When highways are empty and moving at 65 mph, attention concentrates heavily on the diagonal self-loop (alpha_ii approx 0.85). In free-flow, speed is strictly persistent, and upstream sensors have near-zero influence."),
        ("Morning Rush-Hour (08:00 AM - Left):", "When bottlenecks form during morning commute, attention automatically diffuses across upstream and downstream neighbors (alpha_ij approx 0.25 to 0.40). The model actively tracks congestion shockwaves moving backward along the freeway!"),
        ("Empirical Confirmation of Traffic Physics:", "Proves that ST-GAT learns real physical traffic flow principles without hardcoded manual rules.")
    ]
    for h, desc in heat_items:
        p = tf6.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    fig_att_path = FIG_DIR / "evaluation/phase2_attention_weights_heatmap.png"
    if fig_att_path.exists():
        add_card(s6, Inches(6.2), Inches(1.4), Inches(6.3), Inches(5.6), bg_color=RGBColor(255, 255, 255))
        s6.shapes.add_picture(str(fig_att_path), Inches(6.35), Inches(1.75), width=Inches(6.0))

    s6.notes_slide.notes_text_frame.text = (
        "[MEMBER 3 - ANSHIKA (SLIDE 6 - ATTENTION HEATMAP)]\n\n"
        "\"Ab examiners ke samne hamare deep model ka sabse bada visual proof: 'Dynamic Attention Heatmap'.\n\n"
        "Black-box neural networks ke comparison mein Graph Attention Network fully interpretable hota hai. Slide ke right side par "
        "hamara generated heatmap dekhiye: Right side par raat ke 2:00 baje (Midnight Free Flow) ka attention weight matrix hai. "
        "Yahan attention 85% diagonal self-loop par concentrated hai, kyunki jab road khali hoti hai toh traffic speed persistent rehti hai.\n\n"
        "Lekin Left side par subah ke 8:00 baje (Morning Rush Hour) ka matrix dekhiye: Attention weights turant neighbor sensors par diffuse "
        "ho jate hain (alpha_ij = 0.25 se 0.40). Jab highway par bottleneck banta hai, toh peeche aane wali gaadiyon ki speed girne lagti hai—ise "
        "backward shockwave kehte hain. Hamara ST-GAT in shockwaves ko dynamically track kar raha hai! "
        "Ab main Samriddhi ko invite karti hoon final empirical outcomes aur dashboard present karne ke liye.\""
    )

    # =========================================================================
    # SLIDE 7: PHASE 2 EMPIRICAL OUTCOMES (Member 4: Samriddhi Shah)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Phase 2 Empirical Outcomes: ST-GAT Surpasses Baselines", "MEMBER 4: SAMRIDDHI SHAH", "VERIFIED TEST BENCHMARKS")

    add_card(s7, Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.6))
    tb7 = s7.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(5.0), Inches(5.3))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "Held-Out Test Results (90,149 Instances)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(10)

    out_items = [
        ("15-Minute Lookahead (+3 steps):", "• Persistence: 3.0011 mph | Ridge: 2.8789 mph\n• XGBoost (Phase 1): 2.5909 mph\n• ST-GAT (Phase 2): 2.4124 mph (RMSE 4.58, MAPE 6.18%)\n  --> 6.89% error reduction vs XGBoost"),
        ("30-Minute Lookahead (+6 steps):", "• Persistence: 3.6157 mph | Ridge: 3.6050 mph\n• XGBoost (Phase 1): 3.1668 mph\n• ST-GAT (Phase 2): 2.8945 mph (RMSE 5.51, MAPE 7.82%)\n  --> 8.60% error reduction vs XGBoost"),
        ("60-Minute Lookahead (+12 steps) -- THE BREAKTHROUGH:", "• Persistence: 4.7302 mph | Ridge: 4.9066 mph\n• XGBoost (Phase 1): 4.0083 mph\n• ST-GAT (Phase 2): 3.5821 mph (RMSE 6.79, MAPE 10.21%)\n  --> 10.63% relative error reduction vs XGBoost!"),
        ("Architectural Takeaway:", "At 15m, traffic is local, so XGBoost and ST-GAT are close. At 60m, spatial shockwaves dominate—ST-GAT's dynamic attention prevents error compounding!")
    ]
    for h, desc in out_items:
        p = tf7.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    fig_comp_path = FIG_DIR / "evaluation/phase2_st_gat_comparison.png"
    if fig_comp_path.exists():
        add_card(s7, Inches(6.4), Inches(1.4), Inches(6.1), Inches(5.6), bg_color=RGBColor(255, 255, 255))
        s7.shapes.add_picture(str(fig_comp_path), Inches(6.55), Inches(1.65), width=Inches(5.8))

    s7.notes_slide.notes_text_frame.text = (
        "[MEMBER 4 - SAMRIDDHI SHAH (SLIDE 7 - OUTCOMES)]\n\n"
        "\"Thank you Anshika. Respected professors and examiners, main Samriddhi Shah hoon aur main Phase 2 ke verified test outcomes "
        "present kar rahi hoon.\n\n"
        "Slide 7 par hamara multi-horizon benchmark comparison chart hai jo 90,149 held-out test instances par evaluate kiya gaya hai. "
        "ST-GAT ne har single horizon par Phase 1 ke best XGBoost model ko convincingly beat kiya hai:\n"
        "• 15 minutes par MAE 2.59 se ghat kar 2.41 mph ho gaya (-6.89% error reduction).\n"
        "• 30 minutes par MAE 3.17 se ghat kar 2.89 mph ho gaya (-8.60% error reduction).\n"
        "• Aur sabse bada breakthrough 60-minute lookahead par aaya: MAE 4.01 se gir kar 3.58 mph ho gaya—pura 10.63% relative error reduction!\n\n"
        "Yeh result traffic theory ko perfectly justify karta hai: Short horizon par local speed dominate karti hai, isliye XGBoost aur ST-GAT "
        "paas the. Lekin 60 minutes par distant shockwaves poore corridor mein travel karti hain. Tabular model wahan fail hone lagta hai, "
        "jabki hamara dynamic GAT shockwaves ko capture karke error compounding ko rok deta hai.\""
    )

    # =========================================================================
    # SLIDE 8: FORECAST TRACE & STREAMLIT DASHBOARD (Member 4: Samriddhi Shah)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Continuous Forecast Traces & Offline Streamlit App", "MEMBER 4: SAMRIDDHI SHAH")

    fig_trace_path = FIG_DIR / "evaluation/phase2_forecast_trace.png"
    if fig_trace_path.exists():
        add_card(s8, Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.6), bg_color=RGBColor(255, 255, 255))
        s8.shapes.add_picture(str(fig_trace_path), Inches(0.95), Inches(1.75), width=Inches(5.7))

    add_card(s8, Inches(7.0), Inches(1.4), Inches(5.5), Inches(5.6))
    tb8 = s8.shapes.add_textbox(Inches(7.2), Inches(1.5), Inches(5.1), Inches(5.3))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "Offline Interactive Dashboard (app/streamlit_app.py)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    app_items = [
        ("Sub-2-Second Cold Start:", "Operates offline-first using precomputed evaluation parquet files and CSV metrics. Bootstraps instantly on Streamlit Community Cloud without requiring expensive GPU retraining."),
        ("Module 1 - Executive Scope & Lineage:", "Cryptographic data lineage (SHA-256), sensor statistics, and formal academic guardrail compliance."),
        ("Module 2 - Missingness & Sensor Explorer:", "Time-series visualizer across all 207 sensors with missing data filters."),
        ("Module 3 - Multi-Horizon Forecast Lab:", "Users pick any sensor and timestamp to inspect side-by-side ground truth vs predictions across 15m, 30m, and 60m."),
        ("Module 4 - Peak vs Off-Peak Analytics:", "Stratified rush-hour inflation analysis and residual error distributions."),
        ("Module 5 - Network Topology & Phase 2 Outcomes:", "Geographic Caltrans sensor array map, dynamic attention heatmaps, and ST-GAT test metrics.")
    ]
    for h, desc in app_items:
        p = tf8.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(7)

    s8.notes_slide.notes_text_frame.text = (
        "[MEMBER 4 - SAMRIDDHI SHAH (SLIDE 8 - TRACE & APP)]\n\n"
        "\"Slide 8 ke left side par hamara 24-hour continuous speed forecast trace plot hai (+15 min lookahead). "
        "Aap dekh sakte hain ki subah aur shaam ke rush hours mein jab speed achanak 65 mph se girkar 25 mph hoti hai, "
        "hamara ST-GAT model ground truth (black line) ko bina artificial lag ya oversmoothing ke closely track karta hai.\n\n"
        "Examiners aur faculty ke independent verification ke liye humne ek production-grade Streamlit application (app/streamlit_app.py) "
        "deploy kiya hai. Yeh app offline-first architecture follow karta hai: isme precomputed test evaluation parquet cached hai, "
        "jiski wajah se yeh 2 second ke andar boot ho jata hai bina retrain kiye. Isme 5 comprehensive modules hain jahan examiner kisi bhi "
        "sensor aur timestamp ko choose karke real-time forecast aur attention heatmap inspect kar sakte hain.\""
    )

    # =========================================================================
    # SLIDE 9: CONCLUSION & DEFENSE READINESS (Member 4: Samriddhi Shah / All)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Project Conclusion & Committee Defense Verification", "MEMBER 4 / ALL MEMBERS", "FINAL SCIENTIFIC VERIFICATION")

    add_card(s9, Inches(0.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb9_1 = s9.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True

    p = tf9_1.paragraphs[0]
    p.text = "Complete Academic Deliverables"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    d_items = [
        ("IEEE Technical Report:", "Publication-grade conference paper in imp_DOCS/Phase2_IEEE_Technical_Report.pdf and LaTeX in reports/paper/."),
        ("Master Presentation Deck:", "16:9 widescreen presentation in imp_DOCS/Phase2_ST_GAT_Presentation.pptx."),
        ("Interactive Streamlit App:", "Offline app in app/streamlit_app.py ready for deployment."),
        ("Trained Model Checkpoint:", "Serialized PyTorch model at models/st_gat_best_checkpoint.pt."),
        ("Automated Test Suite:", "10/10 passing pytest unit tests in tests/ validating splits, attention, and loss masking.")
    ]
    for h, desc in d_items:
        p = tf9_1.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s9, Inches(4.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb9_2 = s9.shapes.add_textbox(Inches(5.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True

    p = tf9_2.paragraphs[0]
    p.text = "Verified Empirical Takeaways"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(8)

    e_items = [
        ("Phase 1 Baselines:", "Established competitive tabular ML baseline (XGBoost 2.59 mph @ 15m; 4.01 mph @ 60m)."),
        ("Spatial Shockwave Proof:", "Adding 1st-order neighbor lags cut 60m MAE by 5.2% (4.02 -> 3.81 mph)."),
        ("Phase 2 Deep Leap:", "ST-GAT outperformed XGBoost across all lookaheads, achieving 2.41 mph @ 15m and 3.58 mph @ 60m (-10.63% error reduction!)."),
        ("Shockwave Attention:", "Dynamic attention heatmaps confirm spatial diffusion during congestion breakdowns.")
    ]
    for h, desc in e_items:
        p = tf9_2.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    add_card(s9, Inches(8.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb9_3 = s9.shapes.add_textbox(Inches(9.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf9_3 = tb9_3.text_frame
    tf9_3.word_wrap = True

    p = tf9_3.paragraphs[0]
    p.text = "Academic Integrity & Attribution"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    a_items = [
        ("Scientific Integrity:", "Operated strictly within physical speed boundaries (mph) with zero false claims of municipal signal actuation."),
        ("Git Attribution Matrix:", "Every single module and script committed to GitHub 'main' under respective member handles (ankit35027, Yashpal-2004, anshika292005, samriddhi-techy)."),
        ("Viva Defense Guide:", "Comprehensive 20-question examiner defense script documented in VIVA_GUIDE.md."),
        ("Committee Defense Readiness:", "Our 4-member research team welcomes questions from the examining committee!")
    ]
    for h, desc in a_items:
        p = tf9_3.add_paragraph()
        p.text = f"• {h}\n  {desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s9.notes_slide.notes_text_frame.text = (
        "[MEMBER 4 / ALL MEMBERS (SLIDE 9 - CONCLUSION & Q&A)]\n\n"
        "\"To conclude: Hamari four-member engineering team ne project ke dono phases ko rigorous scientific standards ke sath complete kiya hai.\n\n"
        "• Phase 1 mein humne verified tabular ML baselines aur spatial ablation proof establish kiya.\n"
        "• Phase 2 mein humne end-to-end Spatio-Temporal Graph Attention Network implement karke 60-minute MAE ko 4.01 se ghata kar "
        "3.58 mph kar diya—pura 10.63% error drop!\n"
        "• Hamare saare deliverables—IEEE paper, interactive Streamlit dashboard, PyTorch model checkpoints, automated pytest suites, "
        "aur comprehensive viva guide—hamare GitHub repository ke main branch par available hain aur har member ke account se properly committed hain.\n\n"
        "Hum pure project mein academic honesty ke sath khade hain—sirf verified highway speed prediction, bina kisi ungrounded signal control claim ke. "
        "Now, our team is ready and welcomes any questions from the examining committee. Thank you!\""
    )

    out_ppt = REPO_ROOT / "presentation/AML_Full_Project_Phase1_and_Phase2.pptx"
    out_imp = IMP_DOCS_DIR / "Phase2_ST_GAT_Presentation.pptx"
    prs.save(out_ppt)
    prs.save(out_imp)
    print(" Attractive 9-Slide Presentation successfully generated at:")
    print(f"  1. {out_ppt}")
    print(f"  2. {out_imp}")

if __name__ == "__main__":
    create_deck()
