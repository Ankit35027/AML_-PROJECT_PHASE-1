"""
Generates an executive, visual, and highly attractive 9-slide PowerPoint presentation
for Historical METR-LA Traffic Speed Forecasting (Phase 1 Baselines & Phase 2 ST-GAT).
Features:
- Concise, punchy bullet points (no long walls of text!)
- High-impact KPI stat cards and visual badges
- Large embedded high-resolution figures
- Natural Hinglish speaker scripts in Presenter Notes
- Exactly 1 dedicated Phase 1 summary slide (Slide 3)
- 4-member attribution: each member presents 2 to 3 slides
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

# Modern Executive Color Palette
NAVY_HEADER = RGBColor(15, 23, 42)       # #0F172A (Deep Slate Header)
CARD_BG = RGBColor(248, 250, 252)        # #F8FAFC (Clean Light Surface)
CARD_BORDER = RGBColor(226, 232, 240)    # #E2E8F0 (Subtle Border)
TEXT_DARK = RGBColor(15, 23, 42)         # #0F172A (Primary Text)
TEXT_MUTED = RGBColor(100, 116, 139)     # #64748B (Secondary / Muted Text)
TEXT_LIGHT = RGBColor(248, 250, 252)     # #F8FAFC (Header Text)
BLUE_ACCENT = RGBColor(37, 99, 235)      # #2563EB (Electric Blue)
GREEN_ACCENT = RGBColor(16, 185, 129)    # #10B981 (Emerald Green for ST-GAT Wins)
AMBER_ACCENT = RGBColor(245, 158, 11)    # #F59E0B (Guardrail / Warning)
RED_ACCENT = RGBColor(239, 68, 68)       # #EF4444 (Negative comparison)

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, presenter_tag, category="METR-LA TRAFFIC SPEED FORECASTING • 4-MEMBER RESEARCH TEAM"):
        # Top banner background
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = NAVY_HEADER
        top_bar.line.fill.background()

        # Category pill text
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(8.5), Inches(0.3))
        p_cat = tb_cat.text_frame.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(8.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = RGBColor(148, 163, 184)

        # Title text
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(8.5), Inches(0.65))
        p_title = tb_title.text_frame.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_LIGHT

        # Presenter badge pill
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.6), Inches(0.32), Inches(2.9), Inches(0.52))
        badge.fill.solid()
        badge.fill.fore_color.rgb = BLUE_ACCENT
        badge.line.fill.background()
        p_badge = badge.text_frame.paragraphs[0]
        p_badge.text = presenter_tag
        p_badge.font.size = Pt(10)
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

    # 4 Team Member Horizontal Cards
    members_data = [
        ("Ankit Garg", "ankit35027", "WS1: Data & Lineage", "Data Pipeline, Missingness Audit & Lineage Hashes", BLUE_ACCENT),
        ("Yashpal", "Yashpal-2004", "WS2: Lags & GAT Layer", "Causal Windows, Ridge ML & GAT Math Formulation", NAVY_HEADER),
        ("Anshika", "anshika292005", "WS3: Temporal & Heads", "XGBoost, GRU Backbone & Multi-Horizon Heads", BLUE_ACCENT),
        ("Samriddhi Shah", "samriddhishah25@gmail.com", "WS4: Training & App", "Training Harness, Streamlit App & Viva Defense", GREEN_ACCENT),
    ]

    for i, (name, handle, role, desc, color) in enumerate(members_data):
        x = Inches(0.8 + i * 2.98)
        card = add_card(s1, x, Inches(1.4), Inches(2.8), Inches(3.6), bg_color=CARD_BG)
        
        # Color accent strip on card
        strip = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.4), Inches(2.8), Inches(0.1))
        strip.fill.solid()
        strip.fill.fore_color.rgb = color
        strip.line.fill.background()

        tb = s1.shapes.add_textbox(x + Inches(0.15), Inches(1.6), Inches(2.5), Inches(3.2))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = name
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK

        p2 = tf.add_paragraph()
        p2.text = f"@{handle}"
        p2.font.size = Pt(9)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(8)

        p3 = tf.add_paragraph()
        p3.text = role
        p3.font.size = Pt(11)
        p3.font.bold = True
        p3.font.color.rgb = color
        p3.space_after = Pt(6)

        p4 = tf.add_paragraph()
        p4.text = desc
        p4.font.size = Pt(10)
        p4.font.color.rgb = TEXT_DARK

    # Bottom Guardrail Banner (Full Width)
    guardrail = add_card(s1, Inches(0.8), Inches(5.2), Inches(11.73), Inches(1.8), bg_color=RGBColor(254, 243, 199), border_color=AMBER_ACCENT)
    tb_g = s1.shapes.add_textbox(Inches(1.0), Inches(5.3), Inches(11.3), Inches(1.6))
    tf_g = tb_g.text_frame
    tf_g.word_wrap = True

    pg1 = tf_g.paragraphs[0]
    pg1.text = "⚠️ ACADEMIC SCOPE & SCIENTIFIC GUARDRAIL (STRICT TRUTHFULNESS)"
    pg1.font.size = Pt(12)
    pg1.font.bold = True
    pg1.font.color.rgb = RGBColor(180, 83, 9)
    pg1.space_after = Pt(4)

    bullet_g = [
        "Freeway Speed Prediction (mph) Only: METR-LA contains strictly inductive loop detector speed records across Los Angeles freeways.",
        "Zero Signal Claims: Dataset has NO traffic signal phases, cycle timings, or queue lengths. We make zero false claims of live signal control.",
        "Two-Phase Methodology: Phase 1 establishes classical ML baselines; Phase 2 introduces Spatio-Temporal Graph Attention Networks (ST-GAT)."
    ]
    for b in bullet_g:
        p = tf_g.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(2)

    s1.notes_slide.notes_text_frame.text = (
        "[MEMBER 1 - ANKIT GARG]\n"
        "Good morning respected professors. Mera naam Ankit Garg hai aur main apne team members Yashpal, Anshika, aur Samriddhi ke sath hamara project present kar raha hoon.\n"
        "Project shuru karne se pehle sabse important academic guardrail: METR-LA dataset freeway speed (mph) record karta hai, signal timing ya queues record nahi karta. Isliye hum strictly speed forecasting par focus kar rahe hain, bina kisi false signal control claim ke. Hamari team ke 4 members ne dedicated workstreams deliver kiye hain jo hamare repo par committed hain."
    )

    # =========================================================================
    # SLIDE 2: METR-LA DATASET & DATA INTEGRITY (Member 1: Ankit Garg)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "METR-LA Dataset & Data Integrity Protocol", "MEMBER 1: ANKIT GARG")

    # 4 Stat Cards Across Top
    kpis = [
        ("207", "Detectors", "LA Freeway Network", BLUE_ACCENT),
        ("34,272", "Timestamps", "5-Min Steps (4 Months)", NAVY_HEADER),
        ("8.11%", "Zero Speeds", "Sensor Dropouts (0 NaNs)", AMBER_ACCENT),
        ("90,149", "Test Cases", "Held-Out Test Observations", GREEN_ACCENT),
    ]
    for i, (val, unit, sub, color) in enumerate(kpis):
        x = Inches(0.8 + i * 2.98)
        add_card(s2, x, Inches(1.4), Inches(2.8), Inches(1.3))
        tb_k = s2.shapes.add_textbox(x + Inches(0.1), Inches(1.45), Inches(2.6), Inches(1.1))
        tf_k = tb_k.text_frame
        p_v = tf_k.paragraphs[0]
        p_v.text = val
        p_v.font.size = Pt(22)
        p_v.font.bold = True
        p_v.font.color.rgb = color
        p_u = tf_k.add_paragraph()
        p_u.text = f"{unit} • {sub}"
        p_u.font.size = Pt(9.5)
        p_u.font.color.rgb = TEXT_MUTED

    # Bottom Left Card: Data Provenance & Missingness
    add_card(s2, Inches(0.8), Inches(2.9), Inches(5.7), Inches(4.1))
    tb2_l = s2.shapes.add_textbox(Inches(1.0), Inches(3.0), Inches(5.3), Inches(3.9))
    tf2_l = tb2_l.text_frame
    tf2_l.word_wrap = True

    p = tf2_l.paragraphs[0]
    p.text = "Data Provenance & Missingness Protocol"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts2_l = [
        ("Source:", "Caltrans District 7 (I-5, I-10, I-405, US-101), Mar 1 – Jun 27, 2012."),
        ("Provenance:", "Verified SHA-256 hash (64784b76...) logged in data_lineage.json."),
        ("Missingness:", "0 NaNs, 0 impossible speeds (>100 mph), exactly 575,302 zeros (8.11%)."),
        ("Feature Protocol:", "Historical lags use causal forward-fill to handle dropouts."),
        ("Target Protocol:", "Ground truth targets strictly require speed > 0.0 mph for honest evaluation.")
    ]
    for h, b in pts2_l:
        p = tf2_l.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # Bottom Right Card: Chronological Splits (Zero Leakage)
    add_card(s2, Inches(6.8), Inches(2.9), Inches(5.73), Inches(4.1))
    tb2_r = s2.shapes.add_textbox(Inches(7.0), Inches(3.0), Inches(5.3), Inches(3.9))
    tf2_r = tb2_r.text_frame
    tf2_r.word_wrap = True

    p = tf2_r.paragraphs[0]
    p.text = "Strict Chronological 70/10/20 Partitioning"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts2_r = [
        ("No Random K-Fold:", "Random split leaks future speeds into past. Strictly forbidden!"),
        ("Train Set (70%):", "Mar 1 – May 23 (23,990 steps). Scalers fitted ONLY here."),
        ("Val Set (10%):", "May 23 – Jun 4 (3,427 steps). Dedicated to early stopping."),
        ("Test Set (20%):", "Jun 4 – Jun 27 (6,855 steps; 90,149 instances). Frozen benchmark."),
        ("Unit Tested:", "Validated by 6/6 passing unit tests in tests/test_splits_and_leakage.py.")
    ]
    for h, b in pts2_r:
        p = tf2_r.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s2.notes_slide.notes_text_frame.text = (
        "[MEMBER 1 - ANKIT GARG]\n"
        "METR-LA dataset mein LA County ke 207 sensors ka 4 mahine ka data hai (34,272 timestamps). Data audit mein 8.11% zero readings mili jo sensor dropouts thi. Humne causal forward-fill use kiya features ke liye aur valid readings (>0 mph) evaluate ki. Sabse zaroori: humne chronological 70/10/20 split kiya taaki 0% future leakage ho. Ab Yashpal Phase 1 aur GAT explain karenge."
    )

    # =========================================================================
    # SLIDE 3: PHASE 1 SUMMARY (Member 2: Yashpal) -- DEDICATED 1-PAGE SUMMARY
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Phase 1 Summary: Classical ML & Spatial Discovery", "MEMBER 2: YASHPAL", "DEDICATED PHASE 1 EMPIRICAL MILESTONE")

    # 3 Column Cards
    # Card 1: Classical Pipeline
    add_card(s3, Inches(0.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb3_1 = s3.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf3_1 = tb3_1.text_frame
    tf3_1.word_wrap = True

    p = tf3_1.paragraphs[0]
    p.text = "1. Causal ML Pipeline"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts3_1 = [
        ("Causal Window:", "Lags t-11..t (past 60 min) + Sin/Cos Time of Day & Day of Week."),
        ("Horizons:", "+15 min (offset 3), +30 min (offset 6), +60 min (offset 12)."),
        ("Persistence:", "Naive baseline carrying y_t forward: y_hat(t+h) = y_t."),
        ("Ridge Regression:", "L2-regularized linear model with alpha tuned on val set."),
        ("XGBoost Regressor:", "100 gradient boosted trees with validation early stopping.")
    ]
    for h, b in pts3_1:
        p = tf3_1.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # Card 2: Benchmark Results
    add_card(s3, Inches(4.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb3_2 = s3.shapes.add_textbox(Inches(5.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf3_2 = tb3_2.text_frame
    tf3_2.word_wrap = True

    p = tf3_2.paragraphs[0]
    p.text = "2. Test Benchmark Results"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts3_2 = [
        ("15-Min Horizon (+3):", "Persistence 3.00 | Ridge 2.88 | XGBoost 2.59 mph (Best)"),
        ("30-Min Horizon (+6):", "Persistence 3.62 | Ridge 3.61 | XGBoost 3.17 mph (Best)"),
        ("60-Min Horizon (+12):", "Persistence 4.73 | Ridge 4.91 | XGBoost 4.01 mph (Best)"),
        ("Collinearity Finding:", "Ridge degrades worse than Persistence at 60m due to collinear lag extrapolation."),
        ("Rush-Hour Inflation:", "Errors surge 50-65% during peak stop-and-go shockwaves.")
    ]
    for h, b in pts3_2:
        p = tf3_2.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # Card 3: Spatial Ablation Breakthrough
    add_card(s3, Inches(8.8), Inches(1.4), Inches(3.7), Inches(5.6), border_color=GREEN_ACCENT)
    tb3_3 = s3.shapes.add_textbox(Inches(9.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf3_3 = tb3_3.text_frame
    tf3_3.word_wrap = True

    p = tf3_3.paragraphs[0]
    p.text = "3. Spatial Breakthrough"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(8)

    pts3_3 = [
        ("The Experiment:", "Isolated sensor lags vs. adding top-3 road network neighbors."),
        ("15m MAE:", "2.59 mph -> 2.55 mph (-1.5%)"),
        ("30m MAE:", "3.17 mph -> 3.07 mph (-3.2%)"),
        ("60m MAE:", "4.02 mph -> 3.81 mph (-5.2% relative error reduction!)"),
        ("Physical Takeaway:", "Proves traffic shockwaves propagate spatially downstream!"),
        ("Bridge to Phase 2:", "Static neighbor lags helped, but shockwaves travel dynamically -> motivates GAT!")
    ]
    for h, b in pts3_3:
        p = tf3_3.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s3.notes_slide.notes_text_frame.text = (
        "[MEMBER 2 - YASHPAL]\n"
        "Slide 3 hamare Phase 1 ka complete summary hai: Humne causal window se 12 lags liye aur teen models test kiye: Persistence, Ridge, aur XGBoost. XGBoost ne best accuracy di (2.59 mph @ 15m; 4.01 mph @ 60m). Sabse bada breakthrough hamara Spatial Ablation tha: Top-3 road neighbors add karne par 60m error 4.02 se ghatkar 3.81 mph ho gaya (-5.2%). Isne prove kiya ki traffic jams highway par shockwaves ki tarah move karte hain, jo Phase 2 GAT ki foundation bana."
    )

    # =========================================================================
    # SLIDE 4: PHASE 2 DYNAMIC GAT FORMULATION (Member 2: Yashpal)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Phase 2 Formulation: Dynamic Graph Attention", "MEMBER 2: YASHPAL")

    # Left Card: Why Tabular ML Fails
    add_card(s4, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.6))
    tb4_l = s4.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf4_l = tb4_l.text_frame
    tf4_l.word_wrap = True

    p = tf4_l.paragraphs[0]
    p.text = "Why Classical Tabular ML Hits a Ceiling"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RED_ACCENT
    p.space_after = Pt(10)

    pts4_l = [
        ("❌ Static Feature Weights:", "Tabular models assign fixed weights to neighbors. But traffic shockwaves travel dynamically (10–25 mph upstream)."),
        ("❌ Time-Invariant Receptive Field:", "An upstream sensor matters during peak rush hour, but is totally irrelevant at 2 AM. Tabular models cannot dynamically switch focus."),
        ("❌ Loss of Graph Geometry:", "Concatenating columns flattens the highway into a 2D table, ignoring directed roads, split ramps, and road distances."),
        ("⚡ The Deep Solution:", "Compute dynamic edge attention alpha_ij(t) at every timestamp based on real-time traffic conditions!")
    ]
    for h, b in pts4_l:
        p = tf4_l.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    # Right Card: GAT Math Formulation
    add_card(s4, Inches(6.8), Inches(1.4), Inches(5.73), Inches(5.6))
    tb4_r = s4.shapes.add_textbox(Inches(7.0), Inches(1.5), Inches(5.3), Inches(5.3))
    tf4_r = tb4_r.text_frame
    tf4_r.word_wrap = True

    p = tf4_r.paragraphs[0]
    p.text = "Dynamic Spatial Graph Attention (GAT)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    pts4_r = [
        ("Attention Energy e_ij:", "e_ij = LeakyReLU( a^T [ W h_i || W h_j ] )\nProjects sensor features and computes pair-wise dynamic alignment."),
        ("Adjacency-Masked Softmax:", "alpha_ij = exp(e_ij) / sum_{k in N_i} exp(e_ik)\nNormalized strictly across connected road network neighbors N_i."),
        ("Multi-Head Attention (K=4):", "Four independent heads capture diverse traffic phenomena (regular commute vs incident shockwaves)."),
        ("Conservation Law:", "sum_j alpha_ij = 1.0 guaranteed by unit tests in tests/test_phase2_gat.py.")
    ]
    for h, b in pts4_r:
        p = tf4_r.add_paragraph()
        p.text = f"• {h}\n  {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    s4.notes_slide.notes_text_frame.text = (
        "[MEMBER 2 - YASHPAL]\n"
        "XGBoost acha baseline tha lekin tabular models static hote hain: ek upstream sensor jam mein important hota hai par raat ko irrelevant. Tabular ML feature weights ko time ke sath change nahi kar sakta. Isliye Phase 2 mein humne Multi-Head Graph Attention Network formulate kiya. Yeh LeakyReLU aur road adjacency masking se dynamic attention weights alpha_ij calculate karta hai jo har sensor ke neighbors par strictly 1.0 sum karte hain. Ab Anshika architecture explain karengi."
    )

    # =========================================================================
    # SLIDE 5: ST-GAT ARCHITECTURE & TRAINING (Member 3: Anshika)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "ST-GAT Architecture & Training Pipeline", "MEMBER 3: ANSHIKA")

    # 4 Visual Step Flow Blocks Across Middle
    flow_steps = [
        ("1. Input Batch", "(B, 12, 207, 5)", "Speeds + Sin/Cos Time", NAVY_HEADER),
        ("2. Causal GRU", "64 Hidden Units", "Temporal Sequence Encoding", BLUE_ACCENT),
        ("3. 4-Head GAT", "Dynamic Attention", "Spatial Shockwave Routing", BLUE_ACCENT),
        ("4. Direct Heads", "Multi-Horizon", "Direct +15m, +30m, +60m Speeds", GREEN_ACCENT)
    ]
    for i, (title, sub, detail, color) in enumerate(flow_steps):
        x = Inches(0.8 + i * 2.98)
        add_card(s5, x, Inches(1.4), Inches(2.8), Inches(1.4))
        tb_f = s5.shapes.add_textbox(x + Inches(0.1), Inches(1.45), Inches(2.6), Inches(1.3))
        tf_f = tb_f.text_frame
        p_t = tf_f.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = color
        p_s = tf_f.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(11)
        p_s.font.bold = True
        p_s.font.color.rgb = TEXT_DARK
        p_d = tf_f.add_paragraph()
        p_d.text = detail
        p_d.font.size = Pt(9)
        p_d.font.color.rgb = TEXT_MUTED

    # Bottom Left: Model Footprint
    add_card(s5, Inches(0.8), Inches(3.0), Inches(5.7), Inches(4.0))
    tb5_l = s5.shapes.add_textbox(Inches(1.0), Inches(3.1), Inches(5.3), Inches(3.8))
    tf5_l = tb5_l.text_frame
    tf5_l.word_wrap = True

    p = tf5_l.paragraphs[0]
    p.text = "Model Specifications (70,403 Parameters)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts5_l = [
        ("GRU Backbone:", "Causal recurrent unit summarizes past 12 steps into hidden state."),
        ("Spatial GAT:", "4-head attention over sensor adjacency matrix A in R^{207x207}."),
        ("Direct Heads:", "No autoregressive loops -> zero error compounding over time!"),
        ("Lightweight:", "70,403 trainable parameters -> fast inference and low memory."),
        ("Checkpoint:", "Serialized best weights saved at models/st_gat_best_checkpoint.pt.")
    ]
    for h, b in pts5_l:
        p = tf5_l.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # Bottom Right: Optimization Suite
    add_card(s5, Inches(6.8), Inches(3.0), Inches(5.73), Inches(4.0))
    tb5_r = s5.shapes.add_textbox(Inches(7.0), Inches(3.1), Inches(5.3), Inches(3.8))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True

    p = tf5_r.paragraphs[0]
    p.text = "Optimization & Masked Loss Harness"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts5_r = [
        ("Masked MAE Loss:", "L = sum(|y - y_hat| * M) / sum(M)\nBackprops ONLY on valid physical speeds (>0 mph)."),
        ("Sensor Dropouts Ignored:", "Zero speeds (8.11%) produce zero gradients -> model not distracted."),
        ("AdamW Optimizer:", "Learning rate 1e-3, weight decay 1e-4, gradient clipping 5.0."),
        ("Learning Rate Schedule:", "ReduceLROnPlateau halves LR on validation stagnation."),
        ("Validation Early Stopping:", "Monitors validation MAE to prevent overfitting.")
    ]
    for h, b in pts5_r:
        p = tf5_r.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s5.notes_slide.notes_text_frame.text = (
        "[MEMBER 3 - ANSHIKA]\n"
        "Hamara SpatioTemporalGAT model teen parts se bana hai: Causal GRU temporal backbone (64 hidden units), 4-Head Spatial GAT layer, aur Direct Multi-Horizon prediction heads. Pura model sirf 70,403 parameters ka hai, jo lightweight aur fast hai. Training mein humne custom Masked MAE loss lagaya jo sensor dropouts ko ignore karke sirf valid speeds (>0 mph) par learn karta hai. Best model checkpoint models/st_gat_best_checkpoint.pt par saved hai."
    )

    # =========================================================================
    # SLIDE 6: ATTENTION HEATMAPS (Member 3: Anshika) -- EMBEDDED FIGURE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Dynamic Spatial Attention Heatmaps: Tracking Shockwaves", "MEMBER 3: ANSHIKA")

    # Left Card: Physical Interpretability
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

    pts6 = [
        ("Full Model Transparency:", "Unlike black-box NNs, GAT exposes exact attention weights alpha_ij assigned by any sensor to its neighbors."),
        ("Midnight Free-Flow (02:00 AM - Right):", "• Attention concentrates 85% on diagonal self-loop (alpha_ii approx 0.85).\n• Speeds are persistent; neighbor cross-talk is near zero."),
        ("Morning Rush-Hour (08:00 AM - Left):", "• Attention diffuses to neighbors (alpha_ij approx 0.25 to 0.40).\n• Model actively tracks congestion shockwaves moving backward along the freeway!"),
        ("Empirical Validation:", "Direct proof that ST-GAT learns physical traffic flow dynamics without hardcoded rules.")
    ]
    for h, b in pts6:
        p = tf6.add_paragraph()
        p.text = f"• {h}\n  {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    # Right: Embedded Heatmap Figure
    fig_att_path = FIG_DIR / "evaluation/phase2_attention_weights_heatmap.png"
    if fig_att_path.exists():
        add_card(s6, Inches(6.2), Inches(1.4), Inches(6.33), Inches(5.6), bg_color=RGBColor(255, 255, 255))
        s6.shapes.add_picture(str(fig_att_path), Inches(6.35), Inches(1.75), width=Inches(6.0))

    s6.notes_slide.notes_text_frame.text = (
        "[MEMBER 3 - ANSHIKA]\n"
        "Slide 6 par hamara generated attention heatmap hai: Raat ke 2:00 baje (right plot) attention 85% self-loop par hota hai kyunki free flow mein speeds persistent hoti hain. Lekin subah 8:00 baje rush hour mein (left plot) attention diffuse hokar neighbor sensors par fail jata hai (alpha_ij = 0.25 se 0.40). Yeh physically prove karta hai ki hamara model freeway congestion shockwaves ko actively track kar raha hai. Ab Samriddhi final results present karengi."
    )

    # =========================================================================
    # SLIDE 7: EMPIRICAL OUTCOMES (Member 4: Samriddhi Shah) -- EMBEDDED FIGURE
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Phase 2 Empirical Outcomes: ST-GAT Surpasses Baselines", "MEMBER 4: SAMRIDDHI SHAH", "VERIFIED TEST BENCHMARKS")

    # Left Card: Numerical Outcomes
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

    pts7 = [
        ("🏆 15-Minute Lookahead (+3 steps):", "• Persistence: 3.00 mph | Ridge: 2.88 mph | XGBoost: 2.59 mph\n• ST-GAT (Ours): 2.4124 mph (RMSE 4.58, MAPE 6.18%)\n  --> 6.89% error reduction vs XGBoost"),
        ("🏆 30-Minute Lookahead (+6 steps):", "• Persistence: 3.62 mph | Ridge: 3.61 mph | XGBoost: 3.17 mph\n• ST-GAT (Ours): 2.8945 mph (RMSE 5.51, MAPE 7.82%)\n  --> 8.60% error reduction vs XGBoost"),
        ("🚀 60-Minute Lookahead (+12 steps) -- BREAKTHROUGH:", "• Persistence: 4.73 mph | Ridge: 4.91 mph | XGBoost: 4.01 mph\n• ST-GAT (Ours): 3.5821 mph (RMSE 6.79, MAPE 10.21%)\n  --> 10.63% relative error reduction vs XGBoost!"),
        ("Why ST-GAT Dominates at 60m:", "At 15m traffic is local; at 60m shockwaves dominate. Dynamic attention prevents error compounding over medium horizons!")
    ]
    for h, b in pts7:
        p = tf7.add_paragraph()
        p.text = f"{h}\n{b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(8)

    # Right: Embedded Comparison Bar Chart
    fig_comp_path = FIG_DIR / "evaluation/phase2_st_gat_comparison.png"
    if fig_comp_path.exists():
        add_card(s7, Inches(6.4), Inches(1.4), Inches(6.13), Inches(5.6), bg_color=RGBColor(255, 255, 255))
        s7.shapes.add_picture(str(fig_comp_path), Inches(6.55), Inches(1.65), width=Inches(5.8))

    s7.notes_slide.notes_text_frame.text = (
        "[MEMBER 4 - SAMRIDDHI SHAH]\n"
        "Slide 7 par hamara held-out test benchmark chart hai (90,149 instances). ST-GAT ne har single horizon par Phase 1 ke best XGBoost model ko beat kiya hai: 15m par MAE 2.59 se girkar 2.41 mph ho gaya (-6.89%), 30m par 3.17 se 2.89 mph (-8.60%), aur sabse bada breakthrough 60m par: MAE 4.01 se girkar 3.58 mph ho gaya—pura 10.63% error reduction! Yeh prove karta hai ki dynamic attention long horizons par error compounding ko rok leta hai."
    )

    # =========================================================================
    # SLIDE 8: FORECAST TRACE & STREAMLIT DASHBOARD (Member 4: Samriddhi Shah)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Continuous Forecast Traces & Offline Streamlit App", "MEMBER 4: SAMRIDDHI SHAH")

    # Left: Embedded 24h Trace Figure
    fig_trace_path = FIG_DIR / "evaluation/phase2_forecast_trace.png"
    if fig_trace_path.exists():
        add_card(s8, Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.6), bg_color=RGBColor(255, 255, 255))
        s8.shapes.add_picture(str(fig_trace_path), Inches(0.95), Inches(1.75), width=Inches(5.7))

    # Right Card: Streamlit Application Details
    add_card(s8, Inches(7.0), Inches(1.4), Inches(5.53), Inches(5.6))
    tb8 = s8.shapes.add_textbox(Inches(7.2), Inches(1.5), Inches(5.1), Inches(5.3))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "Offline Interactive Dashboard (app/streamlit_app.py)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(10)

    pts8 = [
        ("⚡ Sub-2-Second Cold Start:", "Offline-first architecture with cached test evaluation parquet. Boots instantly on Streamlit Community Cloud without GPU retraining."),
        ("Module 1 - Executive Scope:", "SHA-256 lineage, detector counts, and formal academic guardrail compliance."),
        ("Module 2 - Missingness Explorer:", "Time-series visualizer across all 207 sensors with missing data filters."),
        ("Module 3 - Forecast Lab:", "Interactive side-by-side ground truth vs predictions across 15m, 30m, and 60m."),
        ("Module 4 - Peak vs Off-Peak:", "Stratified rush-hour inflation and residual error distributions."),
        ("Module 5 - Topology & Phase 2:", "Live sensor array map, attention heatmaps, and ST-GAT test metrics.")
    ]
    for h, b in pts8:
        p = tf8.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(7)

    s8.notes_slide.notes_text_frame.text = (
        "[MEMBER 4 - SAMRIDDHI SHAH]\n"
        "Slide 8 ke left side par hamara 24-hour speed forecast trace hai (+15m): aap dekh sakte hain ki jab rush hour mein speed 65 se 25 mph girti hai, ST-GAT bina kisi lag ke ground truth ko track karta hai. Examiners ke live testing ke liye humne interactive Streamlit dashboard (app/streamlit_app.py) banaya hai jo cached test data par 2 second mein boot hota hai bina retraining ke. Isme 5 modules hain jahan koi bhi sensor aur timestamp choose karke predictions aur heatmaps check kiye ja sakte hain."
    )

    # =========================================================================
    # SLIDE 9: CONCLUSION & DEFENSE READINESS (Member 4 / All)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Project Conclusion & Committee Defense Readiness", "MEMBER 4 / ALL MEMBERS", "FINAL SCIENTIFIC VERIFICATION")

    # 3 Column Cards
    # Card 1: Deliverables
    add_card(s9, Inches(0.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb9_1 = s9.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf9_1 = tb9_1.text_frame
    tf9_1.word_wrap = True

    p = tf9_1.paragraphs[0]
    p.text = "What We Delivered ('main')"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts9_1 = [
        ("IEEE Paper:", "Publication-grade conference paper in imp_DOCS/Phase2_IEEE_Technical_Report.pdf."),
        ("Master PPTX:", "Concise 9-slide visual deck in imp_DOCS/Phase2_ST_GAT_Presentation.pptx."),
        ("Streamlit App:", "Interactive dashboard in app/streamlit_app.py."),
        ("Model Checkpoint:", "Trained weights in models/st_gat_best_checkpoint.pt."),
        ("Automated Tests:", "10/10 passing pytest tests in tests/ (splits, GAT, loss).")
    ]
    for h, b in pts9_1:
        p = tf9_1.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # Card 2: Empirical Truths
    add_card(s9, Inches(4.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb9_2 = s9.shapes.add_textbox(Inches(5.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf9_2 = tb9_2.text_frame
    tf9_2.word_wrap = True

    p = tf9_2.paragraphs[0]
    p.text = "Verified Empirical Truths"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GREEN_ACCENT
    p.space_after = Pt(8)

    pts9_2 = [
        ("Phase 1 Baselines:", "XGBoost established competitive classical ML baseline (2.59 mph @ 15m; 4.01 mph @ 60m)."),
        ("Spatial Shockwaves:", "Top-3 neighbor lags cut 60m error by 5.2% (4.02 -> 3.81 mph)."),
        ("Phase 2 Deep Leap:", "ST-GAT achieved 2.41 mph @ 15m and 3.58 mph @ 60m (10.63% error reduction!)."),
        ("Zero Fabrication:", "100% reproducible metrics from test set execution.")
    ]
    for h, b in pts9_2:
        p = tf9_2.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    # Card 3: Defense & Attribution
    add_card(s9, Inches(8.8), Inches(1.4), Inches(3.7), Inches(5.6))
    tb9_3 = s9.shapes.add_textbox(Inches(9.0), Inches(1.5), Inches(3.3), Inches(5.3))
    tf9_3 = tb9_3.text_frame
    tf9_3.word_wrap = True

    p = tf9_3.paragraphs[0]
    p.text = "Defense Readiness"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLUE_ACCENT
    p.space_after = Pt(8)

    pts9_3 = [
        ("Academic Guardrail:", "Strictly freeway speed prediction (mph). Zero claims of live traffic signal control."),
        ("Git Attribution:", "All code pushed to GitHub 'main' under respective member handles (ankit35027, Yashpal-2004, anshika292005, samriddhi-techy)."),
        ("Viva Defense Guide:", "20-question examiner guide documented in VIVA_GUIDE.md."),
        ("Q&A Readiness:", "Our 4-member research team welcomes questions from the committee!")
    ]
    for h, b in pts9_3:
        p = tf9_3.add_paragraph()
        p.text = f"• {h} {b}"
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(6)

    s9.notes_slide.notes_text_frame.text = (
        "[MEMBER 4 / ALL MEMBERS]\n"
        "Conclusion: Hamari 4-member team ne dono phases ko highest scientific integrity ke sath deliver kiya hai. Phase 1 mein verified ML baselines aur spatial ablation diya, aur Phase 2 mein ST-GAT se 60m par 10.63% error reduction achieve kiya. Hamare saare deliverables—IEEE paper, 9-slide deck, Streamlit app, model checkpoints, aur tests—GitHub 'main' par committed hain. Now our entire team is ready for questions from the committee. Thank you!"
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
