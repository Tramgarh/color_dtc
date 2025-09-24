import streamlit as st
import streamlit.components.v1 as components
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from svgpathtools import parse_path
from shapely.geometry import Polygon
import math
import io
import os
import signal
import re

st.title("🎨 SVG Color Detector")


uploaded_file = st.file_uploader("Upload an SVG file", type="svg")



if st.button("❌ Stop App"):
    os.kill(os.getpid(), signal.SIGTERM)



# ---- AREA HELPERS ----
def path_area(d):
    path = parse_path(d)
    # Approximate with 200 samples
    points = [path.point(t/200.0) for t in range(201)]
    poly = Polygon([(p.real, p.imag) for p in points])
    return abs(poly.area)

def rect_area(x, y, w, h):
    return w * h

def circle_area(r):
    return math.pi * (r ** 2)

def ellipse_area(rx, ry):
    return math.pi * rx * ry

def polygon_area(points_str):
    points = [(float(x), float(y)) for x, y in 
              (p.split(',') for p in points_str.strip().split())]
    poly = Polygon(points)
    return abs(poly.area)



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

def px2cm(px_area, dpi=96):
    """Convert pixel² area to cm²."""
    px_to_cm = 2.54 / dpi
    return px_area * (px_to_cm ** 2)

def px2mm(px_area, dpi=96):
    """Convert pixel² area to mm²."""
    cm2 = px2cm(px_area, dpi)
    return cm2 * 100  # 1 cm² = 100 mm²

def get_svg_size():
    """Get the width and height from the SVG file."""
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    viewBox = root.attrib.get("viewBox")
    if width and height:
        return float(width), float(height)
    elif viewBox:
        parts = viewBox.split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])
    return None, None

# ---- XML Parsing ----
img_parse = ET.parse(uploaded_file)
root = img_parse.getroot()
namespace = {"svg": root.tag.split('}')[0].strip('{') if '}' in root.tag else "http://www.w3.org/2000/svg"}
all_elements = root.findall(".//svg:*", namespace)

def xml_color_detection():
    uploaded_file.seek(0)
    if not uploaded_file.read(1):
        st.error("Uploaded file is empty.")
        return None
    uploaded_file.seek(0)

    try:
       
        # Get full image size
        width, height = get_svg_size()
        total_image_area = width * height  # Full image area in pixels

        # Collect the color and area information
        color_area_map = {}

        # Build CSS map from <style> blocks
        css_fill_map = {}
        for elem in root.iter():
            if elem.tag.endswith("style") and elem.text:
                css_text = elem.text.strip()
                rules = re.findall(r"([.#]?[a-zA-Z0-9_-]+)\s*\{([^}]*)\}", css_text)
                for selector, body in rules:
                    fill_match = re.search(r"fill\s*:\s*([^;]+)", body)
                    if fill_match:
                        css_fill_map[selector.strip()] = fill_match.group(1).strip()

        # Collect colors and areas
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]

            # Priority cascade to find the fill color
            fill = elem.attrib.get("fill")
            style = elem.attrib.get("style")
            style_fill = None
            if style:
                for obj in style.split(";"):
                    obj = obj.strip().lower()
                    if obj.startswith("fill:"):
                        style_fill = obj.split(":")[1].strip()

            css_fill = None
            cls = elem.attrib.get("class")
            if cls and ("." + cls in css_fill_map):
                css_fill = css_fill_map["." + cls]

            color = fill or style_fill or css_fill
            if not color or color.lower() in ["none", "white", "#ffffff", "#fff"]:
                continue
            
            # Calculate the area based on the tag type
            area = 0
            if tag == "path" and "d" in elem.attrib:
                area = path_area(elem.attrib["d"])
            elif tag == "rect":
                w = float(elem.attrib.get("width", 0))
                h = float(elem.attrib.get("height", 0))
                area = rect_area(0, 0, w, h)
            elif tag == "circle":
                r = float(elem.attrib.get("r", 0))
                area = circle_area(r)
            elif tag == "ellipse":
                rx = float(elem.attrib.get("rx", 0))
                ry = float(elem.attrib.get("ry", 0))
                area = ellipse_area(rx, ry)
            elif tag == "polygon" and "points" in elem.attrib:
                area = polygon_area(elem.attrib["points"])

            if area > 0:
                color_area_map[color] = color_area_map.get(color, 0) + area

            # -------------------------------
            # ✅ Build dataframe OUTSIDE loop
            # -------------------------------
            if not color_area_map:
                st.error("No colors detected in SVG.")
                return None

            colors_df = pd.DataFrame(list(color_area_map.items()), columns=["color", "total_area"])
            colors_df["percentage"] = (colors_df["total_area"] / total_image_area) * 100
            # Convert area to cm² and mm²
            colors_df['cm2'] = px2cm(colors_df['total_area'])
            colors_df['mm2'] = px2mm(colors_df['total_area'])

            # Add RGB values
            def parse_color(c):
                c = c.strip().lower()
                if c.startswith("rgb"):
                    nums = re.findall(r"\d+", c)
                    return tuple(map(int, nums[:3])) if len(nums) >= 3 else None
                return hex_to_rgb(c.lstrip("#"))  # try hex
            colors_df["RGB"] = colors_df["color"].apply(parse_color)

            return colors_df

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
            st.write(color_count[['color', 'total_area', 'percentage', 'cm2', 'mm2', 'RGB']])
            
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
