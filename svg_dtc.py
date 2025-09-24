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

# ---- AREA HELPERS ----
def path_area(d):
    path = parse_path(d)
    points = [path.point(t / 200.0) for t in range(201)]
    poly = Polygon([(p.real, p.imag) for p in points])
    return abs(poly.area)

def rect_area(x, y, w, h):
    return w * h

def circle_area(r):
    return math.pi * (r ** 2)

def ellipse_area(rx, ry):
    return math.pi * rx * ry

def polygon_area(points_str):
    points = [(float(x), float(y)) for x, y in (p.split(',') for p in points_str.strip().split())]
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
    if rgb is None:
        return "white"
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "black" if luminance > 128 else "white"

def px2cm(px_area, dpi=96):
    px_to_cm = 2.54 / dpi
    return px_area * (px_to_cm ** 2)

def px2mm(px_area, dpi=96):
    cm2 = px2cm(px_area, dpi)
    return cm2 * 100

def get_svg_size(root):
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

def xml_color_detection(file):
    file.seek(0)
    if not file.read(1):
        st.error("Uploaded file is empty.")
        return None
    file.seek(0)

    try:
        img_parse = ET.parse(file)
        root = img_parse.getroot()
        width, height = get_svg_size(root)
        if width is None or height is None:
            st.error("Could not determine SVG dimensions.")
            return None
        total_image_area = width * height

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

        # Collect all colors and areas
        color_area_map = {}
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
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

            area = 0
            try:
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
            except (ValueError, KeyError) as e:
                st.warning(f"Skipping invalid element: {e}")
                continue

            if area > 0:
                color_area_map[color] = color_area_map.get(color, 0) + area

        if not color_area_map:
            st.error("No colors detected in SVG.")
            return None

        # Build DataFrame outside the loop
        colors_df = pd.DataFrame(list(color_area_map.items()), columns=["color", "total_area"])
        colors_df["percentage"] = (colors_df["total_area"] / total_image_area) * 100
        colors_df['cm2'] = colors_df["total_area"].apply(px2cm)
        colors_df['mm2'] = colors_df["total_area"].apply(px2mm)
        colors_df["RGB"] = colors_df["color"].apply(lambda c: parse_color(c) if parse_color(c) else hex_to_rgb(c.lstrip("#")))

        return colors_df

    except ET.ParseError:
        st.error("There was an error parsing the SVG file. Please upload a valid SVG.")
        return None

def parse_color(color):
    color = color.strip().lower()
    if color.startswith("rgb"):
        nums = re.findall(r"\d+", color)
        return tuple(map(int, nums[:3])) if len(nums) >= 3 else None
    return None

def render_svg(file):
    file.seek(0)
    svg_data = file.read().decode("utf-8")
    file.seek(0)
    st.markdown(f'<div style="text-align: center;">{svg_data}</div>', unsafe_allow_html=True)

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
        uploaded_file.seek(0)
        color_count = xml_color_detection(uploaded_file)
        
        if color_count is not None:
            top_colors = color_count.head(10)
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
