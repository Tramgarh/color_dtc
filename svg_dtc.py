import streamlit as st
import streamlit.components.v1 as components
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import io
import os
import signal
import re

st.title("🎨 SVG Color Detector")


uploaded_file = st.file_uploader("Upload an SVG file", type="svg")


def hex_to_rgb(hex_code):
    try:
        hex_code = hex_code.lstrip('#').strip().lower()
        if len(hex_code) == 3:  # Handle shorthand hex (e.g., #fff)
            hex_code = ''.join(c * 2 for c in hex_code)
        if len(hex_code) != 6 or not all(c in '0123456789abcdef' for c in hex_code):
            return None
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    except:
        return None

def get_text_color(rgb):
    """Calculate text color (black or white) based on background brightness."""
    if rgb is None:
        return "white"
    # Luminance formula: 0.299R + 0.587G + 0.114B
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "black" if luminance > 128 else "white"

def xml_color_detection(file):
    file.seek(0)
    if not file.read(1):
        st.error("Uploaded file is empty.")
        return None
    file.seek(0)

    try:
        img_parse = ET.parse(file)
        root = img_parse.getroot()
        namespace = {"svg": root.tag.split('}')[0].strip('{') if '}' in root.tag else "http://www.w3.org/2000/svg"}
        all_elements = root.findall(".//svg:*", namespace)
        colors_list = []
        potential_backgrounds = []

        svg_width = root.attrib.get('width', '100%')
        svg_height = root.attrib.get('height', '100%')
        viewbox = root.attrib.get('viewBox', '').split()
        viewbox_dims = [float(viewbox[2]), float(viewbox[3])] if len(viewbox) == 4 else None

        for el in all_elements:
            # --- Internal CSS ---
            if el.tag.endswith("style") and el.text:
                css_text = el.text.strip()
                rules = re.findall(r"([.#]?[a-zA-Z0-9_-]+)\s*\{([^}]*)\}", css_text)
                for selector, body in rules:
                    fill_match = re.search(r"fill\s*:\s*([^;]+)", body)
                    if fill_match:
                        fill_value = fill_match.group(1).strip()
                        colors_list.append(fill_value)

            # --- Check rect background ---
            is_background = False
            if el.tag.endswith('rect'):
                width = el.attrib.get('width', '')
                height = el.attrib.get('height', '')
                x = el.attrib.get('x', '0')
                y = el.attrib.get('y', '0')
                try:
                    if viewbox_dims:
                        if (width in ['100%', str(viewbox_dims[0])] or 
                            height in ['100%', str(viewbox_dims[1])]) and x == '0' and y == '0':
                            is_background = True
                    elif (width in ['100%', svg_width] or height in ['100%', svg_height]) and x == '0' and y == '0':
                        is_background = True
                except (ValueError, TypeError):
                    pass    

            # --- Inline style ---
            style = el.attrib.get("style")
            if style:
                for obj in style.split(";"):
                    obj = obj.strip().lower()
                    if obj.startswith("fill:"):
                        color = obj.split(":")[1].strip()
                        if color not in ['#ffffff', 'white', '#fff', 'none', 'transparent']:
                            colors_list.append(color)
                            if is_background:
                                potential_backgrounds.append(color)

            # --- Direct fill attr ---
            fill = el.attrib.get('fill')
            if fill and fill.lower() not in ['#ffffff', '#fff', 'white', 'none', 'transparent']:
                colors_list.append(fill)
                if is_background:
                    potential_backgrounds.append(fill)

        # -------------------------------
        # ✅ Build dataframe outside loop
        # -------------------------------
        if not colors_list:
            st.error("No colors detected in SVG.")
            return None

        colors_df = pd.DataFrame(colors_list, columns=["color"])
        color_count = colors_df['color'].value_counts().to_frame("count")
        total = color_count["count"].sum()
        color_count["percentage"] = (color_count["count"] / total) * 100
        color_count.reset_index(inplace=True)

        # ✅ Only keep valid hex values
        color_count['RGB'] = color_count['color'].apply(hex_to_rgb)
        color_count = color_count[color_count['RGB'].notnull()]

        if color_count.empty:
            st.error("No valid hex colors remain after filtering.")
            return None

        if potential_backgrounds:
            st.warning(f"Potential background colors (from full-size rects): {set(potential_backgrounds)}")

        return color_count

    except ET.ParseError:
        st.error("There was an error parsing the SVG file. Please upload a valid SVG.")
        return None

def render_svg(file):
    file.seek(0)
    svg_data = file.read().decode("utf-8")
    file.seek(0)  # Reset for other functions
    st.markdown(f'<div>{svg_data}</div>', unsafe_allow_html=True)

def plot_color_palette_interactive(color_count):
    fig = px.bar(
        color_count,
        x="percentage",
        y=["Colors"] * len(color_count),
        color="color",
        text="color",
        orientation="h",
        color_discrete_sequence=color_count["color"].tolist()
    )

    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="<b>%{text}</b><br>Percentage: %{x:.1f}%",
    )

    fig.update_layout(
        showlegend=False,
        yaxis=dict(showticklabels=False),
        xaxis=dict(title="Percentage", range=[0, 100]),
        height=200
    )

    st.plotly_chart(fig, use_container_width=True)

if uploaded_file:
    if uploaded_file.name.lower().endswith('.svg'):
        render_svg(uploaded_file)
        uploaded_file.seek(0)  # Reset file pointer
        color_count = xml_color_detection(uploaded_file)
        
        if color_count is not None:
            top_colors = color_count.head(10)  # Compute top_colors once
            st.write(color_count[['color', 'count', 'percentage', 'RGB']])
            
            output = io.BytesIO()
            color_count.to_excel(output, index=False, engine="openpyxl")
            st.download_button(
                "📥 Download Excel",
                data=output.getvalue(),
                file_name="color_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_colors["color"], top_colors["percentage"], color=top_colors["color"])
            ax.set_xlabel("Percentage (%)")
            ax.set_title("Color Distribution (Excluding White)")
            ax.invert_yaxis()
            st.pyplot(fig)
            st.title("🎨 Interactive Palette")
            plot_color_palette_interactive(top_colors)

    else:
        st.error("Please upload an SVG file!")

