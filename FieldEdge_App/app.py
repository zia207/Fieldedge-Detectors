"""Field Edge Detector — draw a box on Google / Esri satellite imagery and detect agricultural field
boundaries with models trained on labelled Google-imagery screenshots from Bangladesh.

Run:  cd FieldEdge_App && ../.venv/bin/streamlit run app.py
"""
import base64, hashlib, io, json, math, os, re, tempfile, time, zipfile
from datetime import date, timedelta

import cv2
import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.element import MacroElement
from folium.elements import JSCSSMixin
from folium.plugins import Draw, Fullscreen
from folium.template import Template
from shapely.geometry import box, shape
from streamlit_folium import st_folium

import detector as D

st.set_page_config(page_title="Field Edge Detector", page_icon="🌾", layout="wide")

# FIELD_EDGE_CLOUD=1 is the public Streamlit Community Cloud app: one model, a small box, CPU.
CLOUD = os.environ.get("FIELD_EDGE_CLOUD") == "1"

PAGE_DETECT = "Detect fields"
PAGE_GUIDE = "How to run this app"
PAGE_REPORT = "Model training report"
if CLOUD:
    MAX_KM2 = {"Delineate Anything V2 (fine-tuned) — fast": 0.05}
    MODEL_INFO = {
        "Delineate Anything V2 (fine-tuned) — fast":
            "Fine-tuned YOLO11x-seg. Held-out test: field F1 0.64, boundary F1 0.65 at 1 m. "
            "This public app runs on CPU, so keep the box under 0.05 km². The first visit downloads the weights.",
    }
else:
    MAX_KM2 = {
        "Delineate Anything V2 (fine-tuned) — fast": D.DIRECT_KM2,
        "SAM 2.1 (pre-trained, calibrated) — most accurate": D.DIRECT_KM2,
        "FC-DenseNet — boundary segmentation": D.DIRECT_KM2,
    }
    MODEL_INFO = {
        "Delineate Anything V2 (fine-tuned) — fast":
            "FTW Delineate Anything V2 (YOLO11x-seg) fine-tuned on 36 labelled tiles. Held-out test: field F1 0.64, "
            "boundary F1 0.65 @ 1 m. About 1–2 s per 0.05 km² on GPU.",
        "SAM 2.1 (pre-trained, calibrated) — most accurate":
            "SAM 2.1 Hiera-L automatic mask generation with thresholds calibrated on validation sites. Held-out test: "
            "field F1 0.68, boundary F1 0.77 @ 1 m. About 45 s per 0.05 km² on GPU.",
        "FC-DenseNet — boundary segmentation":
            "FC-DenseNet from models/fc_densenet.py, trained on the labelled tiles to mark boundaries. "
            "A field is a region those boundaries enclose. Validation chooses the depth and the threshold.",
    }
TILE_ATTR = {
    "Google Satellite": "Imagery © Google",
    "Esri World Imagery": "Imagery © Esri, Maxar, Earthstar Geographics",
    D.SENTINEL2: "Sentinel-2 cloudless by EOX — contains modified Copernicus Sentinel data",
}


@st.cache_resource(show_spinner="Loading Delineate Anything V2 (fine-tuned)…")
def get_da():
    return D.load_delineate_anything()


@st.cache_resource(show_spinner="Loading SAM 2.1 Hiera-Large…")
def get_sam2():
    return D.load_sam2()


@st.cache_resource(show_spinner="Loading FC-DenseNet…")
def get_fcdn():
    return D.load_fc_densenet()


class _EdgeEdit(JSCSSMixin, MacroElement):
    """ArcGIS-style line editing on the training GeoTIFF.

    Leaflet.draw's edit tool throws on Leaflet 1.9, so vertex handles come from
    Leaflet.Editable. The toolbar lives in the map: a Streamlit widget would
    rebuild the iframe and drop the undo stack.
    """

    default_js = [(
        "leaflet_editable_js",
        "https://cdnjs.cloudflare.com/ajax/libs/leaflet-editable/1.2.0/Leaflet.Editable.min.js",
    )]
    _template = Template("""
        {% macro html(this, kwargs) %}
        <style>
        .leaflet-vertex-icon {
            background: #fff !important;
            border: 2px solid #e10600 !important;
            border-radius: 50%;
            width: 12px !important;
            height: 12px !important;
            margin-left: -6px !important;
            margin-top: -6px !important;
        }
        .leaflet-middle-icon {
            background: #e10600 !important;
            border: 0 !important;
            border-radius: 50%;
            width: 8px !important;
            height: 8px !important;
            margin-left: -4px !important;
            margin-top: -4px !important;
            opacity: 0.9;
        }
        .fe-editbar {
            background: #fff;
            border: 1px solid #b0b0b0;
            border-radius: 2px;
            padding: 4px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
            width: 214px;
            max-height: 430px;
            overflow: auto;
            box-shadow: 0 1px 4px rgba(0,0,0,.28);
            font: 12px/1.2 Arial, sans-serif;
        }
        .fe-edit-title {
            grid-column: 1 / -1;
            font-weight: 700;
            padding: 2px 4px 4px;
            color: #323232;
        }
        .fe-editbar button {
            border: 1px solid #c8c8c8;
            background: #f7f7f7;
            color: #222;
            padding: 4px 6px;
            text-align: left;
            cursor: pointer;
            border-radius: 2px;
        }
        .fe-editbar button.on { background: #0079c1; color: #fff; border-color: #005e95; }
        .fe-editbar button:hover { background: #e6f2fa; }
        .fe-editbar button.on:hover { background: #006baa; color: #fff; }
        .fe-editbar button.fe-finish {
            grid-column: 1 / -1;
            background: #1e7a46;
            color: #fff;
            border-color: #146336;
            font-weight: 700;
            text-align: center;
        }
        .fe-editbar button.fe-finish:hover { background: #17663a; color: #fff; }
        .fe-editstatus {
            background: rgba(255,255,255,.94);
            padding: 4px 8px;
            font: 12px/1.35 Arial, sans-serif;
            border: 1px solid #ccc;
            max-width: 340px;
            box-shadow: 0 1px 3px rgba(0,0,0,.2);
        }
        </style>
        {% endmacro %}
        {% macro script(this, kwargs) %}
            window.drawnItems = {{ this.group_name }};
            var _edgeMap = {{ this.map_name }};
            if (!_edgeMap.editTools) { _edgeMap.editTools = new L.Editable(_edgeMap); }
            var mode = "vertices";
            var snapOn = true;
            var dirty = false;
            var selected = null;
            var mergeA = null;
            var reshape = null;
            var moving = null;
            var drawingNew = null;
            var undoStack = [];
            var redoStack = [];
            var baseline = "";
            var statusEl = null;
            var bar = null;
            var SNAP_PX = 14;
            var LINE_STYLE = {color: "#e10600", weight: 3, opacity: 0.95, smoothFactor: 0};
            var guide = L.polyline([], {color: "#0079c1", weight: 2, dashArray: "5 4", interactive: false, opacity: 0.95}).addTo(_edgeMap);
            var HELP = {
                select: "Select: click a line to highlight it.",
                vertices: "Vertices: drag a white handle. Click a red midpoint to insert. Right-click a vertex to delete.",
                add: "Add vertex: click a segment to insert a vertex at that spot.",
                delete: "Delete vertex: click a white vertex to remove it. Right-click also deletes.",
                move: "Move: drag a line to shift every vertex. The map does not pan in this tool.",
                continue: "Continue: click an endpoint, click new vertices, then click the last vertex to finish.",
                split: "Split: click a line to break it into two lines.",
                merge: "Merge: click two lines. They join at the closest endpoints.",
                reshape: "Reshape: click the line to start, click the new path, then click the line again to end.",
                newline: "New line: click to add vertices. Click the last vertex to finish the line.",
                delline: "Delete line: click a line to remove it."
            };

            function _edgePush() { _edgeMap.fire("draw:edited", {layers: window.drawnItems}); }
            function geojson() { return JSON.stringify(window.drawnItems.toGeoJSON()); }
            function setStatus(text) { if (statusEl) statusEl.textContent = text; }
            function showHelp() {
                var extra = dirty ? " Unsaved — click Finish to update the shapefile." : "";
                setStatus((HELP[mode] || "") + extra);
            }
            function syncDirty() { dirty = geojson() !== baseline; }
            function markDirty() { redoStack = []; syncDirty(); showHelp(); }
            function latlngsOf(layer) {
                var ll = layer.getLatLngs();
                if (ll.length && ll[0] && typeof ll[0].lat !== "number") ll = ll[0];
                return ll;
            }
            function copyLatLngs(layer) {
                return latlngsOf(layer).map(function (ll) { return L.latLng(ll.lat, ll.lng); });
            }
            function dedupe(lls) {
                var out = [];
                for (var i = 0; i < lls.length; i++) {
                    if (!out.length || Math.abs(out[out.length - 1].lat - lls[i].lat) > 1e-8 || Math.abs(out[out.length - 1].lng - lls[i].lng) > 1e-8) {
                        out.push(lls[i]);
                    }
                }
                return out;
            }
            function paint() {
                window.drawnItems.eachLayer(function (layer) {
                    var on = layer === selected;
                    layer.setStyle({color: on ? "#ffd400" : "#e10600", weight: on ? 5 : 3, opacity: 0.95});
                });
            }
            function select(layer) { selected = layer; paint(); }
            function refreshEdits() {
                var on = mode === "vertices" || mode === "delete";
                window.drawnItems.eachLayer(function (layer) {
                    if (!layer.enableEdit) return;
                    var enabled = layer.editEnabled && layer.editEnabled();
                    if (on && !enabled) layer.enableEdit();
                    if (!on && enabled) layer.disableEdit();
                });
                paint();
            }
            function applyDrag() {
                if (mode === "move") _edgeMap.dragging.disable();
                else _edgeMap.dragging.enable();
            }
            function paintButtons() {
                if (!bar) return;
                var buttons = bar.querySelectorAll("button[data-mode]");
                for (var i = 0; i < buttons.length; i++) {
                    if (buttons[i].getAttribute("data-mode") === mode) L.DomUtil.addClass(buttons[i], "on");
                    else L.DomUtil.removeClass(buttons[i], "on");
                }
            }
            function clearGuide() { guide.setLatLngs([]); }
            function cancelTransient() {
                clearGuide();
                mergeA = null;
                reshape = null;
                if (_edgeMap.editTools.drawing()) _edgeMap.editTools.stopDrawing();
            }
            function checkpoint() {
                var now = geojson();
                if (undoStack.length && undoStack[undoStack.length - 1] === now) return;
                undoStack.push(now);
                if (undoStack.length > 50) undoStack.shift();
                redoStack = [];
            }
            function dropNoopCheckpoint() {
                var now = geojson();
                if (undoStack.length && undoStack[undoStack.length - 1] === now) undoStack.pop();
                else markDirty();
            }
            function addLine(latlngs) {
                var layer = L.polyline(dedupe(latlngs), LINE_STYLE);
                window.drawnItems.addLayer(layer);
                if (mode === "vertices" || mode === "delete") layer.enableEdit();
                return layer;
            }
            function removeLine(layer) {
                if (layer.disableEdit) layer.disableEdit();
                window.drawnItems.removeLayer(layer);
                if (selected === layer) selected = null;
            }
            function setLatLngs(layer, lls) {
                var editing = layer.editEnabled && layer.editEnabled();
                if (editing) layer.disableEdit();
                layer.setLatLngs(dedupe(lls));
                if (editing) layer.enableEdit();
            }
            function restore(fc) {
                cancelTransient();
                window.drawnItems.eachLayer(function (layer) { if (layer.disableEdit) layer.disableEdit(); });
                window.drawnItems.clearLayers();
                (fc.features || []).forEach(function (f) {
                    if (!f.geometry || f.geometry.type !== "LineString") return;
                    addLine(f.geometry.coordinates.map(function (c) { return L.latLng(c[1], c[0]); }));
                });
                selected = null;
                refreshEdits();
            }
            function undo() {
                if (_edgeMap.editTools.drawing()) _edgeMap.editTools.stopDrawing();
                if (!undoStack.length) { setStatus("Nothing to undo."); return; }
                redoStack.push(geojson());
                restore(JSON.parse(undoStack.pop()));
                syncDirty();
                showHelp();
            }
            function redo() {
                if (_edgeMap.editTools.drawing()) _edgeMap.editTools.stopDrawing();
                if (!redoStack.length) { setStatus("Nothing to redo."); return; }
                undoStack.push(geojson());
                restore(JSON.parse(redoStack.pop()));
                syncDirty();
                showHelp();
            }
            function projectOnSegment(latlng, a, b) {
                var P = _edgeMap.latLngToLayerPoint(latlng);
                var A = _edgeMap.latLngToLayerPoint(a);
                var B = _edgeMap.latLngToLayerPoint(b);
                var abx = B.x - A.x, aby = B.y - A.y;
                var den = abx * abx + aby * aby;
                var t = den < 1e-6 ? 0 : ((P.x - A.x) * abx + (P.y - A.y) * aby) / den;
                t = Math.max(0, Math.min(1, t));
                var x = A.x + abx * t, y = A.y + aby * t;
                return {t: t, d: Math.hypot(P.x - x, P.y - y), latlng: _edgeMap.layerPointToLatLng(L.point(x, y))};
            }
            function closestOnLines(latlng, maxPx) {
                var best = null;
                window.drawnItems.eachLayer(function (layer) {
                    var lls = latlngsOf(layer);
                    for (var i = 0; i < lls.length - 1; i++) {
                        var hit = projectOnSegment(latlng, lls[i], lls[i + 1]);
                        if (hit.d <= maxPx && (!best || hit.d < best.d)) {
                            best = {d: hit.d, t: hit.t, i: i, layer: layer, latlng: hit.latlng};
                        }
                    }
                });
                return best;
            }
            function closestEndpoint(latlng, maxPx) {
                var best = null;
                var p = _edgeMap.latLngToLayerPoint(latlng);
                window.drawnItems.eachLayer(function (layer) {
                    var lls = latlngsOf(layer);
                    if (lls.length < 2) return;
                    [0, lls.length - 1].forEach(function (idx) {
                        var d = _edgeMap.latLngToLayerPoint(lls[idx]).distanceTo(p);
                        if (d <= maxPx && (!best || d < best.d)) best = {d: d, layer: layer, atStart: idx === 0};
                    });
                });
                return best;
            }
            function snapTarget(latlng, skipLayer, skipIndex) {
                var p = _edgeMap.latLngToLayerPoint(latlng);
                var best = null;
                window.drawnItems.eachLayer(function (layer) {
                    var lls = latlngsOf(layer);
                    for (var i = 0; i < lls.length; i++) {
                        if (layer === skipLayer && i === skipIndex) continue;
                        var d = _edgeMap.latLngToLayerPoint(lls[i]).distanceTo(p);
                        if (d < 0.5 || d > SNAP_PX) continue;
                        if (!best || d < best.d) best = {d: d, lat: lls[i].lat, lng: lls[i].lng};
                    }
                });
                if (best) return best;
                window.drawnItems.eachLayer(function (layer) {
                    if (layer === skipLayer) return;
                    var lls = latlngsOf(layer);
                    for (var i = 0; i < lls.length - 1; i++) {
                        var hit = projectOnSegment(latlng, lls[i], lls[i + 1]);
                        if (hit.d < 0.5 || hit.d > 10) continue;
                        if (!best || hit.d < best.d) best = {d: hit.d, lat: hit.latlng.lat, lng: hit.latlng.lng};
                    }
                });
                return best;
            }
            function joinLines(a, b) {
                a = a.map(function (ll) { return L.latLng(ll.lat, ll.lng); });
                b = b.map(function (ll) { return L.latLng(ll.lat, ll.lng); });
                var ends = [
                    {d: a[0].distanceTo(b[0]), ra: true, rb: true},
                    {d: a[0].distanceTo(b[b.length - 1]), ra: true, rb: false},
                    {d: a[a.length - 1].distanceTo(b[0]), ra: false, rb: true},
                    {d: a[a.length - 1].distanceTo(b[b.length - 1]), ra: false, rb: false}
                ];
                ends.sort(function (x, y) { return x.d - y.d; });
                var pick = ends[0];
                if (pick.ra) a.reverse();
                if (!pick.rb) b.reverse();
                if (a[a.length - 1].distanceTo(b[0]) < 0.05) return a.concat(b.slice(1));
                return a.concat(b);
            }
            function lineOpts() { return {color: "#e10600", weight: 3, opacity: 0.95, smoothFactor: 0}; }
            function startNew(latlng) {
                drawingNew = _edgeMap.editTools.startPolyline(latlng || null, lineOpts());
            }
            function adopt(layer) {
                var tools = _edgeMap.editTools;
                if (tools.featuresLayer && tools.featuresLayer.hasLayer(layer)) tools.featuresLayer.removeLayer(layer);
                if (!window.drawnItems.hasLayer(layer)) window.drawnItems.addLayer(layer);
                layer.setStyle(LINE_STYLE);
                if (mode === "vertices" || mode === "delete") {
                    if (!layer.editEnabled()) layer.enableEdit();
                } else if (layer.editEnabled && layer.editEnabled()) {
                    layer.disableEdit();
                }
                select(layer);
            }
            function setMode(name) {
                cancelTransient();
                mode = name;
                paintButtons();
                refreshEdits();
                applyDrag();
                showHelp();
                if (name === "newline") startNew(null);
            }
            function toggleSnap() {
                snapOn = !snapOn;
                var btn = bar.querySelector('button[data-action="snap"]');
                if (btn) {
                    if (snapOn) L.DomUtil.addClass(btn, "on");
                    else L.DomUtil.removeClass(btn, "on");
                }
                setStatus(snapOn ? "Snapping on (14 px)." : "Snapping off.");
            }
            function finish() {
                if (_edgeMap.editTools.drawing()) _edgeMap.editTools.commitDrawing();
                if (!dirty) { setStatus("Nothing new to save."); return; }
                _edgePush();
                dirty = false;
                setStatus("Saved. The shapefile download uses these lines.");
            }
            function isUi(e) {
                var t = e.originalEvent && e.originalEvent.target;
                if (!t || !t.closest) return false;
                return !!t.closest(".fe-editbar, .fe-editstatus, .leaflet-control");
            }
            function applyReshape(r, end) {
                var lls = copyLatLngs(r.layer);
                var a = {i: r.i0, t: r.t0, p: r.p0};
                var b = {i: end.i, t: end.t, p: L.latLng(end.latlng.lat, end.latlng.lng)};
                var sketch = r.sketch.map(function (p) { return L.latLng(p.lat, p.lng); });
                if (b.i < a.i || (b.i === a.i && b.t < a.t)) {
                    var tmp = a; a = b; b = tmp; sketch.reverse();
                }
                var head = lls.slice(0, a.i + 1);
                if (a.t > 0.05) head.push(a.p);
                var tail = [];
                if (b.t < 0.95) tail.push(b.p);
                tail = tail.concat(lls.slice(b.i + 1));
                var next = dedupe(head.concat(sketch).concat(tail));
                if (next.length < 2) { setStatus("That reshape is too short."); return; }
                checkpoint();
                setLatLngs(r.layer, next);
                select(r.layer);
                markDirty();
            }
            function endMove() {
                if (!moving) return;
                moving = null;
                dropNoopCheckpoint();
            }

            var EditBar = L.Control.extend({
                options: {position: "topleft"},
                onAdd: function () {
                    bar = L.DomUtil.create("div", "fe-editbar");
                    L.DomEvent.disableClickPropagation(bar);
                    L.DomEvent.disableScrollPropagation(bar);
                    var title = L.DomUtil.create("div", "fe-edit-title", bar);
                    title.textContent = "Edit";
                    var modes = [
                        ["select", "Select"],
                        ["vertices", "Vertices"],
                        ["add", "Add vertex"],
                        ["delete", "Delete vertex"],
                        ["move", "Move"],
                        ["continue", "Continue"],
                        ["split", "Split"],
                        ["merge", "Merge"],
                        ["reshape", "Reshape"],
                        ["newline", "New line"],
                        ["delline", "Delete line"]
                    ];
                    modes.forEach(function (item) {
                        var b = L.DomUtil.create("button", "", bar);
                        b.type = "button";
                        b.setAttribute("data-mode", item[0]);
                        b.textContent = item[1];
                        b.title = HELP[item[0]];
                        L.DomEvent.on(b, "click", function () { setMode(item[0]); });
                    });
                    var actions = [
                        ["snap", "Snapping", "Snap vertices to nearby vertices (14 px). On by default."],
                        ["undo", "Undo", "Undo the last edit (Ctrl+Z)."],
                        ["redo", "Redo", "Redo (Ctrl+Y)."],
                        ["finish", "Finish", "Write the edited lines into the shapefile."]
                    ];
                    actions.forEach(function (item) {
                        var b = L.DomUtil.create("button", item[0] === "finish" ? "fe-finish" : "", bar);
                        b.type = "button";
                        b.setAttribute("data-action", item[0]);
                        b.textContent = item[1];
                        b.title = item[2];
                        if (item[0] === "snap") L.DomUtil.addClass(b, "on");
                        L.DomEvent.on(b, "click", function () {
                            if (item[0] === "snap") toggleSnap();
                            else if (item[0] === "undo") undo();
                            else if (item[0] === "redo") redo();
                            else finish();
                        });
                    });
                    return bar;
                }
            });
            var StatusBar = L.Control.extend({
                options: {position: "bottomright"},
                onAdd: function () {
                    statusEl = L.DomUtil.create("div", "fe-editstatus");
                    L.DomEvent.disableClickPropagation(statusEl);
                    statusEl.textContent = HELP.vertices;
                    return statusEl;
                }
            });
            _edgeMap.addControl(new EditBar());
            _edgeMap.addControl(new StatusBar());

            _edgeMap.on("editable:vertex:rawclick", function (e) {
                if (mode !== "delete") { e._cancelled = true; return; }
                if (e.vertex && e.vertex.latlngs.length <= 2) {
                    e._cancelled = true;
                    setStatus("A line needs at least two vertices.");
                    return;
                }
                checkpoint();
            });
            _edgeMap.on("editable:vertex:contextmenu", function (e) {
                if (e.originalEvent) L.DomEvent.preventDefault(e.originalEvent);
                if (!e.vertex) return;
                if (e.vertex.latlngs.length <= 2) { setStatus("A line needs at least two vertices."); return; }
                checkpoint();
                e.vertex.delete();
            });
            _edgeMap.on("editable:vertex:dragstart", function () { checkpoint(); });
            _edgeMap.on("editable:middlemarker:mousedown", function () { checkpoint(); });
            _edgeMap.on("editable:vertex:dragend", function (e) {
                if (snapOn && e.vertex) {
                    var hit = snapTarget(e.vertex.latlng, e.layer, e.vertex.getIndex());
                    if (hit) {
                        e.vertex.latlng.update(hit);
                        e.vertex._latlng = e.vertex.latlng;
                        e.vertex.setLatLng(e.vertex.latlng);
                        if (e.vertex.editor) e.vertex.editor.refresh();
                    }
                }
                dropNoopCheckpoint();
            });
            _edgeMap.on("editable:vertex:deleted", function () { markDirty(); });
            _edgeMap.on("editable:vertex:new", function (e) {
                if (e.layer && window.drawnItems.hasLayer(e.layer)) markDirty();
            });
            _edgeMap.on("editable:drawing:clicked", function (e) {
                if (!snapOn || !e.layer) return;
                var lls = latlngsOf(e.layer);
                if (!lls.length) return;
                var hit = snapTarget(e.latlng || lls[lls.length - 1], null, -1);
                if (!hit) return;
                var last = lls[lls.length - 1];
                if (last && last.update) last.update(hit);
                e.layer.redraw();
            });
            _edgeMap.on("editable:drawing:commit", function (e) {
                var layer = e.layer;
                var wasNew = !!(layer && layer === drawingNew);
                drawingNew = null;
                if (wasNew && layer) {
                    var n = latlngsOf(layer).length;
                    if (n < 2) {
                        if (_edgeMap.editTools.featuresLayer.hasLayer(layer)) _edgeMap.editTools.featuresLayer.removeLayer(layer);
                        return;
                    }
                    checkpoint();
                    adopt(layer);
                    markDirty();
                    setStatus("Line added. Click Finish to save.");
                    return;
                }
                dropNoopCheckpoint();
            });
            _edgeMap.on("editable:drawing:cancel", function (e) {
                if (!drawingNew) return;
                if (e.layer && e.layer !== drawingNew) return;
                var layer = drawingNew;
                drawingNew = null;
                if (layer.disableEdit) layer.disableEdit();
                if (_edgeMap.editTools.featuresLayer && _edgeMap.editTools.featuresLayer.hasLayer(layer)) {
                    _edgeMap.editTools.featuresLayer.removeLayer(layer);
                }
                if (window.drawnItems.hasLayer(layer)) window.drawnItems.removeLayer(layer);
                if (_edgeMap.hasLayer(layer)) _edgeMap.removeLayer(layer);
            });

            _edgeMap.on("mousedown", function (e) {
                if (mode !== "move" || isUi(e) || moving) return;
                var hit = closestOnLines(e.latlng, 12);
                if (!hit) return;
                checkpoint();
                moving = {layer: hit.layer, from: L.latLng(e.latlng.lat, e.latlng.lng), orig: copyLatLngs(hit.layer)};
                select(hit.layer);
                L.DomEvent.stop(e);
            });
            _edgeMap.on("mousemove", function (e) {
                if (!moving) return;
                var dlat = e.latlng.lat - moving.from.lat;
                var dlng = e.latlng.lng - moving.from.lng;
                moving.layer.setLatLngs(moving.orig.map(function (ll) {
                    return L.latLng(ll.lat + dlat, ll.lng + dlng);
                }));
            });
            _edgeMap.on("mouseup", endMove);
            L.DomEvent.on(window, "mouseup", endMove);

            _edgeMap.on("click", function (e) {
                if (_edgeMap.editTools.drawing() || isUi(e) || mode === "move") return;
                if (mode === "vertices" || mode === "select" || mode === "delete") {
                    var picked = closestOnLines(e.latlng, 14);
                    if (picked) select(picked.layer);
                    else if (mode === "select") { selected = null; paint(); }
                    if (mode === "select") showHelp();
                    return;
                }
                if (mode === "newline") {
                    if (!_edgeMap.editTools.drawing()) startNew(e.latlng);
                    return;
                }
                if (mode === "add") {
                    var addHit = closestOnLines(e.latlng, 14);
                    if (!addHit || addHit.t < 0.03 || addHit.t > 0.97) {
                        setStatus("Click a segment, between its vertices.");
                        return;
                    }
                    checkpoint();
                    var addLls = copyLatLngs(addHit.layer);
                    addLls.splice(addHit.i + 1, 0, L.latLng(addHit.latlng.lat, addHit.latlng.lng));
                    setLatLngs(addHit.layer, addLls);
                    select(addHit.layer);
                    markDirty();
                    setStatus("Vertex added. Click Finish to save.");
                    return;
                }
                if (mode === "delline") {
                    var delHit = closestOnLines(e.latlng, 14);
                    if (!delHit) { setStatus("Click a line to delete it."); return; }
                    checkpoint();
                    removeLine(delHit.layer);
                    paint();
                    markDirty();
                    setStatus("Line deleted. Click Finish to save.");
                    return;
                }
                if (mode === "split") {
                    var splitHit = closestOnLines(e.latlng, 14);
                    if (!splitHit) { setStatus("Click a line to split it."); return; }
                    var lls = copyLatLngs(splitHit.layer);
                    var mid = L.latLng(splitHit.latlng.lat, splitHit.latlng.lng);
                    var a, b;
                    if (splitHit.t <= 0.03) { a = lls.slice(0, splitHit.i + 1); b = lls.slice(splitHit.i); }
                    else if (splitHit.t >= 0.97) { a = lls.slice(0, splitHit.i + 2); b = lls.slice(splitHit.i + 1); }
                    else { a = lls.slice(0, splitHit.i + 1).concat([mid]); b = [mid].concat(lls.slice(splitHit.i + 1)); }
                    if (a.length < 2 || b.length < 2) { setStatus("Click between the ends of the line."); return; }
                    checkpoint();
                    removeLine(splitHit.layer);
                    addLine(a);
                    select(addLine(b));
                    markDirty();
                    setStatus("Line split. Click Finish to save.");
                    return;
                }
                if (mode === "merge") {
                    var mergeHit = closestOnLines(e.latlng, 14);
                    if (!mergeHit) { setStatus("Click the first line."); return; }
                    if (!mergeA) {
                        mergeA = mergeHit.layer;
                        select(mergeA);
                        setStatus("Click the second line. They join at the closest endpoints.");
                        return;
                    }
                    if (mergeA === mergeHit.layer) { setStatus("Click a different line."); return; }
                    var merged = joinLines(latlngsOf(mergeA), latlngsOf(mergeHit.layer));
                    checkpoint();
                    removeLine(mergeA);
                    removeLine(mergeHit.layer);
                    mergeA = null;
                    select(addLine(merged));
                    markDirty();
                    setStatus("Lines merged. Click Finish to save.");
                    return;
                }
                if (mode === "continue") {
                    var end = closestEndpoint(e.latlng, 18);
                    if (!end) { setStatus("Click an endpoint (either end of a line)."); return; }
                    checkpoint();
                    if (!end.layer.editEnabled()) end.layer.enableEdit();
                    select(end.layer);
                    if (end.atStart) end.layer.editor.continueBackward();
                    else end.layer.editor.continueForward();
                    setStatus("Click to add vertices. Click the last vertex to finish.");
                    return;
                }
                if (mode === "reshape") {
                    var hit = closestOnLines(e.latlng, 14);
                    if (!reshape) {
                        if (!hit) { setStatus("Click the line where the reshape should start."); return; }
                        reshape = {layer: hit.layer, i0: hit.i, t0: hit.t, p0: L.latLng(hit.latlng.lat, hit.latlng.lng), sketch: []};
                        select(hit.layer);
                        setStatus("Click the new vertices, then click the line again to finish.");
                        return;
                    }
                    var same = hit && hit.layer === reshape.layer;
                    var far = same && _edgeMap.latLngToLayerPoint(hit.latlng).distanceTo(_edgeMap.latLngToLayerPoint(reshape.p0)) > 12;
                    if (same && far && reshape.sketch.length) {
                        applyReshape(reshape, hit);
                        reshape = null;
                        clearGuide();
                        setStatus("Reshape applied. Click Finish to save.");
                        return;
                    }
                    var pt = e.latlng;
                    if (snapOn) {
                        var s = snapTarget(pt, reshape.layer, -1);
                        if (s) pt = L.latLng(s.lat, s.lng);
                    }
                    reshape.sketch.push(L.latLng(pt.lat, pt.lng));
                    guide.setLatLngs([reshape.p0].concat(reshape.sketch));
                    setStatus("Vertex added to the reshape. Click the line again to finish.");
                }
            });
            L.DomEvent.on(_edgeMap.getContainer(), "keydown", function (ev) {
                var key = ev.key || "";
                if ((ev.ctrlKey || ev.metaKey) && (key === "z" || key === "Z")) {
                    L.DomEvent.preventDefault(ev);
                    if (ev.shiftKey) redo(); else undo();
                } else if ((ev.ctrlKey || ev.metaKey) && (key === "y" || key === "Y")) {
                    L.DomEvent.preventDefault(ev);
                    redo();
                } else if (key === "Escape") {
                    cancelTransient();
                    showHelp();
                }
            });

            setMode("vertices");
            baseline = geojson();
            dirty = false;
            showHelp();
            window._edgeToolsReady = true;
        {% endmacro %}
        """)

    def __init__(self, group_name, map_name):
        super().__init__()
        self._name = "EdgeEdit"
        self.group_name = group_name
        self.map_name = map_name


def _vertex_count(gdf):
    """Coordinate count of a line layer, for the straighten summary."""
    if gdf is None or not len(gdf):
        return 0
    n = 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            n += len(geom.coords)
        elif geom.geom_type == "MultiLineString":
            n += sum(len(part.coords) for part in geom.geoms)
    return n


def _edit_token(features):
    if features is None:
        return "predicted"
    blob = json.dumps(features, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


class _ImageryBrightness(MacroElement):
    """CSS brightness on the tile images and the GeoTIFF overlay, not on the vectors."""

    _template = Template("""
        {% macro header(this, kwargs) %}
        <style>
        #{{ this._parent.get_name() }} .leaflet-tile-pane,
        #{{ this._parent.get_name() }} img.leaflet-image-layer {
            filter: brightness({{ this.factor }});
        }
        </style>
        {% endmacro %}
        """)

    def __init__(self, factor):
        super().__init__()
        self._name = "ImageryBrightness"
        self.factor = f"{float(factor):.2f}"


def _brighten_imagery(mapa, brightness):
    if abs(float(brightness) - 1.0) >= 1e-6:
        _ImageryBrightness(brightness).add_to(mapa)


def _reported_drawing(out):
    """The drawn feature, [] if the map reported an empty drawing, or None if it has not reported."""
    if not out:
        return None
    feat = out.get("last_active_drawing")
    if feat and feat.get("geometry"):
        return feat
    drawings = out.get("all_drawings")
    if drawings is None:
        return None
    if not drawings:
        return []
    last = drawings[-1]
    return last if last and last.get("geometry") else []


def _edge_editor_map(rgb, bounds, features):
    """GeoTIFF image with the training polylines open for vertex editing."""
    west, south, east, north = bounds
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], tiles=None,
                   zoom_start=18, max_zoom=24, min_zoom=12, control_scale=True)
    folium.raster_layers.ImageOverlay(
        image=rgb, bounds=[[south, west], [north, east]], opacity=1, interactive=False, zindex=1, alt="GeoTIFF",
    ).add_to(m)
    group = folium.FeatureGroup(name="Field edges")
    for feat in features:
        coords = feat["geometry"]["coordinates"]
        folium.PolyLine(
            [(y, x) for x, y in coords], color="#e10600", weight=3, opacity=0.95,
            smooth_factor=0, interactive=False,
        ).add_to(group)
    group.add_to(m)
    _EdgeEdit(group.get_name(), m.get_name()).add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m


def _labeled_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Labeled_Data")


def _quality_report_path():
    return os.path.join(_labeled_dir(), "field_boundary_training_report.md")


_FIG_MARK = re.compile(r"<!--\s*fig:(\d+)\s*-->")


@st.cache_data(show_spinner=False)
def _report_figure_bytes(_mtime):
    """PNG figures embedded in the notebook HTML export, in document order."""
    path = os.path.join(_labeled_dir(), "field_boundary_training.html")
    with open(path, encoding="utf-8", errors="replace") as handle:
        html = handle.read()
    blobs = re.findall(r'<img\b[^>]*src="data:image/png;base64,([^"]+)"', html)
    return [base64.b64decode(blob) for blob in blobs]


def _render_guide():
    if CLOUD:
        st.markdown(
            """
Draw a rectangle with the ▭ tool, or jump with **Location** or with latitude and longitude,
then click **Detect field edges**. This public copy runs Delineate Anything on a CPU.
Keep the box under **0.05 km²**. The first visit downloads the weights.

The model was trained on labelled Google Satellite screenshots from 10 sites in Bangladesh,
at about 0.25 m per pixel. On the held-out test sites it reached field F1 0.64 and boundary
F1 0.65 at 1 m. It is strongest on narrow plots in that kind of landscape, on Google Satellite
at the 20 m scale (zoom 19).

Accuracy drops on landscapes and field patterns unlike those sites, and on zoom 18
(about 0.55 m), which is used only when zoom 19 imagery is missing. Google Satellite and
Esri have no date control. Sentinel-2 pixels are 10 m, so narrow plots stay merged.
Google's terms restrict bulk download and reuse of its imagery; use it for research.
Esri World Imagery is the safer source for work you will share.
            """
        )
    else:
        st.markdown(
            """
#### Run it

From the project folder:

```bash
cd FieldEdge_App
../.venv/bin/streamlit run app.py
```

Open http://localhost:8501. Use that virtual environment. A CUDA GPU is what the timings below assume.
SAM 2.1 on CPU is too slow to be useful. Delineate Anything still runs on CPU, slowly.

In the page: choose imagery, jump with **Location** or with latitude and longitude, draw a rectangle
with the ▭ tool, then click **Detect field edges**.

#### Where it performs best

The models were trained on labelled Google Satellite screenshots from 10 sites in Bangladesh,
at about 0.25 m per pixel. They are strongest on narrow plots in that kind of landscape, on
Google Satellite at the 20 m scale (zoom 19).

On the held-out test sites, Delineate Anything reached field F1 0.64 and boundary F1 0.65 at 1 m,
in about 1–2 seconds per 0.05 km² on a GPU. SAM 2.1 reached field F1 0.68 and boundary F1 0.77
at 1 m, in about 45 seconds per 0.05 km². Use Delineate Anything for a quick pass. Use SAM 2.1
when you can wait. The Steger refinement can move an outline by at most 0.75 m onto a dark bund,
and it helps where that bund is visible at sub-metre resolution.

A rectangle up to 4 km² is one detection.

#### Where it performs poorly

Expect weaker outlines on landscapes and field layouts unlike those 10 sites. Zoom 18
(about 0.55 m) is the fallback when zoom 19 is missing, and it sits farther from the training resolution.

Sentinel-2 pixels are 10 m. Narrow-plot mode resamples that picture to 0.25 m and still has only
the original 10 m of detail. Large-field mode runs SAM 2.1 at 2.5 m on one screenshot, about half
a minute per sample tile. Plots narrower than about 20 m stay merged. Adjacent fields of the same
crop on the same date often share a colour, so SAM draws one region. False colour (near-infrared,
red, green) separates more of those fields than true colour does.

FC-DenseNet103 was selected on the validation set: boundary F1 0.34, threshold 0.40, epoch 25.
A held-out test score for it is not reported here.

#### Limits

- Only a rectangle can be selected. From 4 km² to 10 km² the rectangle is split into sample tiles
  of at most 4 km², overlapping by 80 m. Above 10 km², detection is refused.
- Google Satellite and Esri World Imagery are the current mosaic. They have no date.
- Sentinel-2 uses one scene inside **From** and **To**: the clearest scene under the cloud limit.
  Cloud cover is for the whole satellite tile, so a cloudy tile can hide a clear field.
  The map you draw on stays the 2024 true-colour mosaic. The dated picture is the detection image.
  Dates cannot be stepped through, and two seasons cannot be shown side by side.
- False colour is three bands from that one scene. The Fields of the World checkpoint needs two
  dates and eight bands, and this app leaves that checkpoint unused.
- **RGB + NIR (SAM 2.1)** is Sentinel-2 only. It picks the scene true colour would use, downloads
  red, green, blue, and near-infrared, runs SAM 2.1 at 2.5 m, then polygonizes: simplify
  (default 15 m), minimum size (default 500 m²), optional maximum size, and optional hole removal.
- Sharpen strengthens edges about 10 m wide and leaves the pixel size at 10 m.
- Viewer brightness changes the screen. Detection and the downloaded GeoTIFF keep the original pixels.
  Finish a vertex edit before moving that slider, or the unfinished sketch resets.
- Large-field mode turns Steger off, because a 10 m pixel has no sub-metre bund to snap to.
- The 0.25 m models drop fields smaller than about 20 m². In large-field mode the same pixel count
  is about 0.2 ha.
- Google's terms restrict bulk download and reuse of its imagery. Use it for research.
  Esri World Imagery is the safer source for work you will share. Sentinel-2 is Copernicus data;
  keep the attribution on the map.
            """
        )

def _render_report():
    path = _quality_report_path()
    if not os.path.isfile(path):
        st.warning("The training report is not in this copy of the app.")
        return
    with open(path, encoding="utf-8") as report:
        text = report.read()
    html_path = os.path.join(_labeled_dir(), "field_boundary_training.html")
    figures = _report_figure_bytes(os.path.getmtime(html_path)) if os.path.isfile(html_path) else []
    parts = _FIG_MARK.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
            continue
        index = int(part) - 1
        if 0 <= index < len(figures):
            st.image(figures[index], width="stretch")


# ── sidebar ──────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
PYGIS_LOGO = os.path.join(_APP_DIR, "..", "pygis_logo.png")
UPATTA_LOGO = os.path.join(_APP_DIR, "..", "upatta_logo.png")
with open(UPATTA_LOGO, "rb") as _logo_file:
    _upatta_b64 = base64.b64encode(_logo_file.read()).decode()

with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.fe-pygis)
            + [data-testid="stElementContainer"] [data-testid="stImage"] {
            display: flex;
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.fe-pygis)
            + [data-testid="stElementContainer"] img {
            width: 60% !important;
            height: auto !important;
        }
        [data-testid="stSidebar"] { container-type: inline-size; }
        [data-testid="stSidebarContent"] { padding-bottom: 15rem; }
        .fe-brand-footer {
            position: fixed;
            z-index: 1000002;
            left: 0;
            bottom: 0;
            width: 100cqw;
            box-sizing: border-box;
            margin: 0;
            padding: 0.55rem 1.15rem 0.7rem;
            background: rgb(240, 242, 246);
            border-top: 1px solid rgba(49, 51, 63, 0.12);
        }
        .fe-brand-footer p {
            margin: 0 0 0.3rem 0;
            font-size: 0.95rem;
            font-weight: 600;
            line-height: 1.2;
        }
        .fe-brand-footer img {
            display: block;
            width: 11rem;
            height: auto;
        }
        .fe-banner {
            background: #14532d;
            color: #f8faf5;
            border-radius: 10px;
            padding: 0.95rem 1.25rem;
            margin: 0 0 1rem 0;
        }
        .fe-banner h1 {
            margin: 0;
            color: #f8faf5;
            font-size: 1.75rem;
            line-height: 1.2;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fe-pygis"></div>', unsafe_allow_html=True)
    st.image(PYGIS_LOGO, width="stretch")
    page = st.radio(
        "Pages",
        [PAGE_DETECT, PAGE_GUIDE, PAGE_REPORT],
        key="fe_page",
    )
    if page == PAGE_DETECT:
        source = st.selectbox(
            "Imagery", list(D.SOURCES), index=1 if CLOUD else 0,
            help="Google is the imagery the models were trained on. Esri World Imagery is licence-friendlier. "
                 "Sentinel-2 is 10 m Copernicus imagery. False colour is built from the Planetary Computer scene.",
        )
        s2_start, s2_end, s2_cloud, s2_sharpen = None, None, 30.0, 0.0
        s2_colour = D.S2_TRUE
        s2_large = False
        s2_rgbnir = False
        s2_simplify, s2_min_m2, s2_max_m2, s2_close_holes = 15.0, 500.0, 0.0, False
        if source == D.SENTINEL2:
            st.caption("Sentinel-2 L2A at 10 m, one date, from "
                       "[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/). "
                       "The same scene is shown in the "
                       "[Copernicus Browser](https://browser.dataspace.copernicus.eu/). "
                       "The map stays the 2024 true-colour mosaic. Detection uses the colour chosen below.")
            today = date.today()
            s2_start = st.date_input("From", value=today - timedelta(days=365))
            s2_end = st.date_input("To", value=today)
            s2_cloud = st.slider("Maximum cloud cover (%)", 5, 80, 30, 5)
            _colours = [D.S2_FALSE, D.S2_TRUE] if CLOUD else [D.S2_FALSE, D.S2_TRUE, D.S2_RGBNIR]
            s2_colour = st.radio(
                "Scene colour",
                _colours,
                index=0,
                help="False colour places near-infrared, red, and green (bands B08, B04, B03) into the picture. "
                     "Crops show red, so neighbouring fields separate more often than in true colour. "
                     "RGB + NIR picks the same scene as true colour, then downloads red, green, blue, and "
                     "near-infrared and runs SAM 2.1. The map you draw on stays true colour.",
            )
            s2_rgbnir = (not CLOUD) and s2_colour == D.S2_RGBNIR
            if s2_rgbnir:
                s2_large = True
                st.caption("The scene is the one true colour would use. SAM 2.1 then reads near-infrared, red, "
                           "and the average of green and blue, at 2.5 m. Outlines are polygonized afterwards.")
                s2_simplify = st.slider(
                    "Simplify (m)", 0, 40, 15, 1,
                    help="Douglas–Peucker tolerance in metres, the same control as "
                         "ftw inference polygonize --simplify. 15 is the Fields of the World default. 0 keeps every vertex.",
                )
                s2_min_m2 = st.number_input(
                    "Minimum field size (m²)", min_value=0.0, value=500.0, step=100.0,
                    help="Fields smaller than this are dropped. 500 m² is the Fields of the World default. 0 keeps them.",
                )
                s2_max_m2 = st.number_input(
                    "Maximum field size (m²)", min_value=0.0, value=0.0, step=1000.0,
                    help="Fields larger than this are dropped. 0 leaves the upper size open, which is the "
                         "Fields of the World default.",
                )
                s2_close_holes = st.checkbox(
                    "Remove holes inside fields",
                    value=False,
                    help="Drops interior rings, the same as ftw inference polygonize --close_interiors.",
                )
            elif not CLOUD:
                s2_field = st.radio(
                    "Field size",
                    ["Large fields (SAM 2.1 on the screenshot)", "Narrow plots (0.25 m models)"],
                    index=0,
                    help="Large fields: one screenshot, resampled to 2.5 m, then SAM 2.1. "
                         "False colour is the better screenshot for large fields. "
                         "The FTW checkpoint needs two dates and eight bands, so a screenshot cannot drive it. "
                         "Narrow plots: the models trained at 0.25 m. They miss fields that fill the scene.",
                )
                s2_large = s2_field.startswith("Large")
            s2_sharpen = st.slider(
                "Sharpen", 0.0, 3.0, 1.5, 0.5,
                help="Unsharp mask on the 10 m scene, before it is resampled. "
                     "0 leaves the scene unchanged. This strengthens edges about 10 m wide and does not add finer detail.",
            )
            zoom = D.NATIVE_ZOOM[D.SENTINEL2]
        else:
            scale = st.radio("Map scale / resolution", list(D.SCALES), index=0)
            zoom = D.SCALES[scale]
        brightness = st.slider(
            "Viewer brightness", 0.5, 2.0, 1.0, 0.1,
            key="viewer_brightness",
            help="Lightens or darkens the map, the captured preview, and the GeoTIFF editor. "
                 "Detection and the downloaded GeoTIFF keep the original brightness. "
                 "Finish a vertex edit before moving this, or the unfinished sketch is reset.",
        )
        if s2_rgbnir:
            model_name = "SAM 2.1 (pre-trained, calibrated) — most accurate"
            st.caption("SAM 2.1 on the four-band scene at 2.5 m. Polygonize then simplifies the outlines "
                       "and applies the size limits.")
        elif s2_large:
            model_name = "SAM 2.1 (pre-trained, calibrated) — most accurate"
            _shot = "false-colour" if s2_colour == D.S2_FALSE else "true-colour"
            st.caption(f"SAM 2.1 on one {_shot} screenshot at 2.5 m. A field can cover most of the scene. "
                       "The 0.25 m models are for the narrow-plot option.")
        elif len(MODEL_INFO) == 1:
            model_name = next(iter(MODEL_INFO))
            st.markdown(f"**Model:** {model_name}")
            st.caption(MODEL_INFO[model_name])
        else:
            model_name = st.radio("Model", list(MODEL_INFO), index=0)
            st.caption(MODEL_INFO[model_name])
        if s2_large:
            refine = False
        else:
            refine = st.checkbox(
                "Refine outlines (Steger 1998 + snake)",
                value=True,
                help="After the model, each outline may move up to 0.75 m onto a dark bund centre. "
                     "Interior Steger lines are saved as a layer and do not split fields.",
            )
        with st.expander("Advanced"):
            da_conf = st.slider("Delineate Anything: confidence threshold", 0.05, 0.60, D.DA_SETTINGS["conf"], 0.05)
            sam_iou = D.SAM2_SETTINGS["pred_iou_thresh"] if CLOUD else st.slider(
                "SAM 2: predicted-IoU threshold", 0.65, 0.95, D.SAM2_SETTINGS["pred_iou_thresh"], 0.05)
            _fcdn = None if CLOUD else D.fcdn_selected()
            fcdn_thr = 0.50 if CLOUD else st.slider(
                "FC-DenseNet: boundary threshold", 0.20, 0.80,
                float(_fcdn["threshold"]) if _fcdn else 0.50, 0.05)
            auto_fallback = False if source == D.SENTINEL2 else st.checkbox(
                "Fall back to zoom 18 if zoom 19 has no imagery", value=True)
        st.markdown("---")
        where = "downloaded on first run" if CLOUD else "`Labeled_Data/`"
        st.caption(f"Compute device: **{D.DEVICE.upper()}** · weights {where}")
        if source == "Google Satellite":
            st.caption("⚠️ Google's Terms of Service restrict bulk download and reuse of imagery. "
                       "For research use only; use Esri World Imagery for shared or operational work.")
        elif source == D.SENTINEL2:
            st.caption("Contains modified Copernicus Sentinel data. Cloud cover is for the whole satellite tile.")
    st.markdown(
        f'''
        <div class="fe-brand-footer">
          <p>Zia Ahmed, PhD</p>
          <img src="data:image/png;base64,{_upatta_b64}" alt="Upatta Data Analytics">
        </div>
        ''',
        unsafe_allow_html=True,
    )

def _banner():
    st.markdown(
        """
        <div class="fe-banner">
          <h1>🌾 Field Edge Detector</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── pages ────────────────────────────────────────────────────────────────
_banner()
if page == PAGE_GUIDE:
    st.session_state._fe_page_shown = PAGE_GUIDE
    st.markdown("### How to run this app, where it works, and its limits")
    _render_guide()
    st.stop()
if page == PAGE_REPORT:
    st.session_state._fe_page_shown = PAGE_REPORT
    _render_report()
    st.stop()

st.markdown(
    "Draw a rectangle on a satellite image and this app outlines the fields inside it. "
    "Choose the imagery and the model in the sidebar, jump to a place by name or by coordinates, "
    "then run detection. The models were trained on labelled screenshots from 10 sites in Bangladesh. "
    "**How to run this app** and **Model training report** open from **Pages** in the sidebar."
)

# ── map + drawing ────────────────────────────────────────────────────────
st.markdown("### 1 · Draw a rectangle around the fields")
if CLOUD:
    _area_caption = f"Maximum area: **{MAX_KM2[model_name]} km²**."
else:
    _area_caption = (
        f"Draw a rectangle with the ▭ tool. Up to **{D.DIRECT_KM2:.0f} km²** is detected as one area. "
        f"A larger rectangle, up to **{D.SELECT_MAX_KM2:.0f} km²**, is split into sample tiles of at most "
        f"{D.DIRECT_KM2:.0f} km². Only a rectangle can be selected."
    )
st.caption("Pan or zoom first, or jump to a location or to coordinates below. " + _area_caption)

def _apply_place(hit):
    st.session_state.center = [hit["lat"], hit["lon"]]
    st.session_state.zoom = int(hit["zoom"])
    st.session_state.lat_in = float(hit["lat"])
    st.session_state.lon_in = float(hit["lon"])

def _go_to_location():
    query = (st.session_state.get("location_query") or "").strip()
    st.session_state.geocode_error = ""
    if not query:
        st.session_state.geocode_hits = []
        st.session_state.geocode_error = "Enter a town, village, or address."
        return
    try:
        hits = D.geocode_location(query)
    except LookupError as exc:
        st.session_state.geocode_hits = []
        st.session_state.geocode_error = str(exc)
        return
    st.session_state.geocode_hits = hits
    st.session_state.geocode_pick = 0
    if hits:
        _apply_place(hits[0])
    else:
        st.session_state.geocode_error = "No place found. Try a town, village, or address."

def _pick_geocode():
    hits = st.session_state.get("geocode_hits") or []
    pick = int(st.session_state.get("geocode_pick") or 0)
    if 0 <= pick < len(hits):
        _apply_place(hits[pick])

c1, c2, c3 = st.columns([1, 1, 1])
if "center" not in st.session_state:
    st.session_state.center = [22.315167, 90.154757]          # Bhola, Bangladesh
if "zoom" not in st.session_state:
    st.session_state.zoom = 17
# Widget values are dropped while another page is open. Put the map centre back
# into the boxes only when this page is opened again, so a typed edit is kept.
if st.session_state.get("_fe_page_shown") != PAGE_DETECT:
    st.session_state.lat_in = float(st.session_state.center[0])
    st.session_state.lon_in = float(st.session_state.center[1])
st.session_state._fe_page_shown = PAGE_DETECT
lat_in = c1.number_input("Latitude", format="%.6f", key="lat_in")
lon_in = c2.number_input("Longitude", format="%.6f", key="lon_in")
if c3.button("Go to coordinates", width="stretch"):
    st.session_state.center = [float(lat_in), float(lon_in)]
    st.session_state.zoom = 17

loc1, loc2 = st.columns([4, 1], vertical_alignment="bottom")
loc1.text_input("Location", key="location_query", placeholder="Town, village, or address")
loc2.button("Go to location", width="stretch", on_click=_go_to_location)
st.caption("Geocoding provided © [OpenStreetMap](https://www.openstreetmap.org/copyright), openstreetmap.org")
if st.session_state.get("geocode_error"):
    st.warning(st.session_state.geocode_error)
geocode_hits = st.session_state.get("geocode_hits") or []
if len(geocode_hits) > 1:
    st.selectbox(
        "Matches",
        range(len(geocode_hits)),
        format_func=lambda i: geocode_hits[i]["name"],
        key="geocode_pick",
        on_change=_pick_geocode,
    )
elif len(geocode_hits) == 1:
    st.caption(geocode_hits[0]["name"])

tiles = D.SOURCES[source].replace("{x}", "{x}").replace("{y}", "{y}").replace("{z}", "{z}")
m = folium.Map(location=st.session_state.center, zoom_start=int(st.session_state.zoom),
               max_zoom=20, tiles=None, control_scale=True)
folium.TileLayer(tiles=tiles, attr=TILE_ATTR[source], name=source, max_zoom=20,
                 max_native_zoom=D.NATIVE_ZOOM[source]).add_to(m)
_brighten_imagery(m, brightness)
Draw(draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False,
                   "circlemarker": False, "rectangle": {"shapeOptions": {"color": "#ffd400", "weight": 2}}},
     edit_options={"edit": False}).add_to(m)
Fullscreen().add_to(m)
saved = st.session_state.get("drawn_feature")
if saved and saved.get("geometry"):
    saved_geom = shape(saved["geometry"])
    if D.is_latlon_rectangle(saved_geom):
        sw, ss, se, sn = saved_geom.bounds
        folium.Rectangle([[ss, sw], [sn, se]], color="#ffd400", weight=2, fill=False).add_to(m)
out = st_folium(m, height=520, use_container_width=True, key=f"draw_{source}",
                returned_objects=["last_active_drawing", "all_drawings"])

def _preview_canvas(bbox, max_side=1400):
    west, south, east, north = bbox
    lat = (south + north) / 2
    width_m = (east - west) * 111320 * math.cos(math.radians(lat))
    height_m = (north - south) * 110570
    scale = max_side / max(width_m, height_m, 1.0)
    return np.zeros((max(1, int(round(height_m * scale))), max(1, int(round(width_m * scale))), 3), np.uint8)


def _paste_preview(canvas, outer, preview, infer, core):
    """Paste the core of a tile preview onto the low-resolution canvas of the whole rectangle."""
    if preview is None or preview.size == 0:
        return
    ph, pw = preview.shape[:2]
    iw, is_, ie, inn = infer
    cw, cs, ce, cn = core
    if abs(ie - iw) < 1e-15 or abs(inn - is_) < 1e-15:
        return
    x0 = int(np.clip(round((cw - iw) / (ie - iw) * pw), 0, pw))
    x1 = int(np.clip(round((ce - iw) / (ie - iw) * pw), 0, pw))
    y0 = int(np.clip(round((inn - cn) / (inn - is_) * ph), 0, ph))
    y1 = int(np.clip(round((inn - cs) / (inn - is_) * ph), 0, ph))
    if x1 <= x0 or y1 <= y0:
        return
    crop = preview[y0:y1, x0:x1]
    H, W = canvas.shape[:2]
    ow, os_, oe, on = outer
    cx0 = int(np.clip(round((cw - ow) / (oe - ow) * W), 0, W))
    cx1 = int(np.clip(round((ce - ow) / (oe - ow) * W), 0, W))
    cy0 = int(np.clip(round((on - cn) / (on - os_) * H), 0, H))
    cy1 = int(np.clip(round((on - cs) / (on - os_) * H), 0, H))
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas[cy0:cy1, cx0:cx1] = cv2.resize(crop, (cx1 - cx0, cy1 - cy0), interpolation=cv2.INTER_AREA)


bbox = None
drawn_geom = None
reported = _reported_drawing(out)
if isinstance(reported, dict) and reported.get("geometry"):
    st.session_state.drawn_feature = reported
drawn = st.session_state.get("drawn_feature")
if drawn and drawn.get("geometry"):
    drawn_geom = shape(drawn["geometry"])
    if D.is_latlon_rectangle(drawn_geom):
        bbox = drawn_geom.bounds

# ── run ──────────────────────────────────────────────────────────────────
st.markdown("### 2 · Detect field edges")
if drawn_geom is not None and bbox is None:
    st.error("Draw a rectangle. A selection larger than "
             f"{D.DIRECT_KM2:.0f} km² is split into sample tiles, so the area has to be rectangular.")
elif bbox is None:
    st.info("Draw a rectangle on the map to enable detection.")
else:
    area = D.bbox_area_km2(*bbox)
    area_max = MAX_KM2[model_name] if CLOUD else D.SELECT_MAX_KM2
    direct = MAX_KM2[model_name] if CLOUD else D.DIRECT_KM2
    sample_tiles = [bbox] if area <= direct + 1e-9 else D.sample_tile_grid(*bbox, direct)
    ok = area <= area_max + 1e-9
    a1, a2, a3 = st.columns(3)
    a1.metric("Box area", f"{area:.3f} km²")
    a2.metric("Box size", f"{(bbox[2]-bbox[0])*111320*np.cos(np.radians((bbox[1]+bbox[3])/2)):.0f} × {(bbox[3]-bbox[1])*110570:.0f} m")
    a3.metric("Sample tiles", f"{len(sample_tiles)}" if ok else "–")
    if not ok:
        st.error(f"The rectangle is {area:.2f} km². The maximum is {area_max:.0f} km². "
                 f"Draw a smaller rectangle. Above {direct:.0f} km² it is split into sample tiles.")
    elif len(sample_tiles) > 1:
        st.info(f"This rectangle is {area:.2f} km², so it will be split into {len(sample_tiles)} sample tiles "
                f"of at most {direct:.0f} km². Each tile is detected on its own, then the fields are merged.")
    if s2_large and ok:
        st.caption(f"SAM 2.1 reads the 10 m screenshot at {D.S2_LARGE_GSD:.1f} m. About half a minute per sample tile.")
    elif (not CLOUD) and model_name.startswith("SAM") and ok and area > 0.25:
        minutes = area / 0.05 * 45 / 60
        st.caption(f"SAM 2.1 on {area:.2f} km² takes roughly {minutes:.0f} minutes on this GPU.")
    if st.button("🔍 Detect field edges", type="primary", disabled=not ok, width="stretch"):
        prog = st.progress(0.0, text="Downloading imagery…"); t0 = time.time()
        n_tiles = len(sample_tiles)
        field_parts, steger_parts = [], []
        captures, notes, missing = [], [], 0
        used_zoom, ground = zoom, None
        scene_labels = []
        preview = _preview_canvas(bbox)
        for i, core in enumerate(sample_tiles):
            infer = D.expand_bbox(core, D.TILE_OVERLAP_M, bbox) if n_tiles > 1 else core

            def tile_progress(frac, msg, i=i, n=n_tiles):
                prog.progress(min(1.0, (i + frac) / n), text=f"Sample tile {i + 1}/{n}: {msg}" if n > 1 else msg)

            tile_progress(0.02, "Downloading imagery…")
            if source == D.SENTINEL2:
                try:
                    rgb, tr, miss, scene = D.capture_sentinel2(
                        *infer, s2_start.isoformat(), s2_end.isoformat(), float(s2_cloud),
                        colour=s2_colour)
                except LookupError as exc:
                    st.error(str(exc))
                    st.stop()
                tile_zoom = D.NATIVE_ZOOM[D.SENTINEL2]
                scene_labels.append(f"{scene['date']} ({scene['cloud']:.0f}% cloud)")
            else:
                rgb, tr, miss = D.capture(*infer, zoom, source)
                tile_zoom = zoom
                if D.placeholder_fraction(rgb) > 0.5 and zoom == 19 and auto_fallback:
                    rgb, tr, miss = D.capture(*infer, 18, source)
                    tile_zoom = 18
                    notes.append(f"{source} has no zoom-19 imagery here; zoom 18 was used instead.")
            if D.placeholder_fraction(rgb) > 0.5:
                if source == D.SENTINEL2:
                    st.error("This Sentinel-2 scene has no data for the rectangle. Try other dates.")
                else:
                    st.error(f"{source} has no imagery for this rectangle at zoom {tile_zoom}. Try the other source.")
                st.stop()
            missing += miss
            used_zoom = tile_zoom
            if source == D.SENTINEL2 and s2_sharpen > 0:
                rgb = D.sharpen_rgb(rgb, s2_sharpen)
            work_gsd = D.S2_LARGE_GSD if s2_large else D.GSD
            img, tr_img, ground = D.to_training_resolution(rgb, tr, (infer[1] + infer[3]) / 2, gsd=work_gsd)
            cb = lambda f, msg, tp=tile_progress: tp(0.12 + 0.62 * f, msg)
            if model_name.startswith("Delineate"):
                try:
                    polys, scores = D.detect_delineate_anything(
                        get_da(), img, patch=D.DA_SETTINGS["patch"], conf=da_conf, progress=cb)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()
            elif model_name.startswith("FC-DenseNet"):
                try:
                    polys, scores = D.detect_fc_densenet(get_fcdn(), img, thr=fcdn_thr, progress=cb)
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.stop()
            else:
                polys, scores = D.detect_sam2(
                    get_sam2(), img, progress=cb, pred_iou_thresh=sam_iou,
                    max_area_frac=0.85 if s2_large else 0.25)
            labels = D.polygons_to_labels(polys, img.shape[:2])
            steger = None
            if refine:
                refined, n_snap, curves = D.refine_outlines(
                    img, labels, progress=lambda f, msg, tp=tile_progress: tp(0.78 + 0.14 * f, msg))
                steger = D.curves_to_gdf(curves, tr_img)
                labels = refined
                if steger is not None and len(steger):
                    steger_parts.append(steger)
            else:
                n_snap = 0
            tile_progress(0.94, "Building polygons…")
            fields_i, _lines_i = D.to_vectors(polys, scores, tr_img, img.shape[:2], box(*infer), labels=labels)
            field_parts.append(D.keep_core_fields(fields_i, core, bbox))
            _paste_preview(preview, bbox, D.outline_preview(img, labels), infer, core)
            captures.append({"bbox": core, "rgb": rgb, "transform": tr, "snapped": n_snap})
        fields = D.concat_fields(field_parts)
        lines = D.lines_from_fields(fields)
        if s2_rgbnir:
            fields, lines = D.polygonize_fields(
                fields, s2_simplify, s2_min_m2, s2_max_m2, s2_close_holes)
        steger = D.concat_gdfs(steger_parts)
        note = " ".join(dict.fromkeys(notes))
        st.session_state.result = dict(
            fields=fields, lines=lines, steger=steger, snapped=sum(c["snapped"] for c in captures),
            refined=refine, preview=preview, image=preview,
            capture=(captures[0]["rgb"], captures[0]["transform"]),
            tiles=captures if n_tiles > 1 else None,
            sample_tiles=[c["bbox"] for c in captures] if n_tiles > 1 else None,
            bbox=bbox, source=source, zoom=used_zoom, ground=ground, model=model_name, note=note,
            scene=" · ".join(dict.fromkeys(scene_labels)),
            colour=s2_colour if source == D.SENTINEL2 else "",
            polygonize=({
                "simplify_m": float(s2_simplify),
                "min_m2": float(s2_min_m2),
                "max_m2": float(s2_max_m2),
                "close_interiors": bool(s2_close_holes),
            } if s2_rgbnir else None),
            sharpen=float(s2_sharpen) if source == D.SENTINEL2 else 0.0,
            work_gsd=D.S2_LARGE_GSD if s2_large else D.GSD,
            missing=missing, seconds=time.time() - t0, run_id=time.strftime("%Y%m%dT%H%M%S"))
        prog.progress(1.0, text="Done")

# ── results ──────────────────────────────────────────────────────────────
res = st.session_state.get("result")
if res:
    straight = bool(st.session_state.get("straighten_edges", False))
    tol_m = float(st.session_state.get("straighten_tol_m", 3.0))
    if straight and len(res["fields"]):
        f, lines = D.straighten_fields(res["fields"], tol_m)
    else:
        f, lines = res["fields"], res["lines"]
    st.markdown("### 3 · Results")
    if res["note"]:
        st.warning(res["note"])
    if res["missing"]:
        st.warning(f"{res['missing']} imagery tiles could not be downloaded (shown black).")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fields detected", f"{len(f)}")
    m2.metric("Median field size", f"{f.area_m2.median():.0f} m²" if len(f) else "–")
    m3.metric("Outlines snapped", f"{res.get('snapped', 0)}" if res.get("refined") else "off")
    m4.metric("Run time", f"{res['seconds']:.1f} s")
    if res["source"] == D.SENTINEL2:
        scale_name = "10 m"
    else:
        scale_name = "20 m" if res["zoom"] == 19 else "50 m" if res["zoom"] == 18 else f"zoom {res['zoom']}"
    scene_bit = f" Scene {res['scene']}." if res.get("scene") else ""
    if res.get("colour") == D.S2_FALSE:
        scene_bit += " False colour (near-infrared, red, green)."
    elif res.get("colour") == D.S2_TRUE:
        scene_bit += " True colour."
    elif res.get("colour") == D.S2_RGBNIR:
        scene_bit += " Red, green, blue, and near-infrared. SAM reads near-infrared, red, and the average of green and blue."
    poly = res.get("polygonize") or None
    if poly:
        scene_bit += f" Polygonized: simplify {poly['simplify_m']:.0f} m, minimum {poly['min_m2']:.0f} m²"
        if poly.get("max_m2"):
            scene_bit += f", maximum {poly['max_m2']:.0f} m²"
        if poly.get("close_interiors"):
            scene_bit += ", holes removed"
        scene_bit += "."
    if res.get("sharpen"):
        scene_bit += f" Unsharp mask {res['sharpen']:.1f} on the 10 m scene."
    st.caption(
        f"Model: {res['model']}. {res['source']} at {scale_name} scale "
        f"({res['ground']:.2f} m/px), resampled to {res.get('work_gsd', D.GSD):.2f} m/px.{scene_bit} "
        + ("Outlines were moved at most 0.75 m onto a Steger line centre. Interior lines were not used to split fields."
           if res.get("refined") else "Outlines are the model masks, without the snake.")
        + (f" Merged from {len(res['sample_tiles'])} sample tiles (cyan outlines)."
           if res.get("sample_tiles") else "")
    )

    left, right = st.columns([3, 2])
    with left:
        b = res["bbox"]
        rm = folium.Map(location=[(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], tiles=None, max_zoom=20, control_scale=True)
        folium.TileLayer(tiles=D.SOURCES[res["source"]], attr=TILE_ATTR[res["source"]], name=res["source"],
                         max_zoom=20, max_native_zoom=D.NATIVE_ZOOM.get(res["source"], 19)).add_to(rm)
        _brighten_imagery(rm, brightness)
        if len(f):
            folium.GeoJson(f, name="Fields", style_function=lambda _: {"color": "#ff2d2d", "weight": 1.5,
                                                                      "fillColor": "#ffd400", "fillOpacity": 0.12},
                           tooltip=folium.GeoJsonTooltip(["field_id", "area_m2", "score"],
                                                         aliases=["Field", "Area (m²)", "Score"])).add_to(rm)
        for tile in res.get("sample_tiles") or []:
            tw, ts, te, tn = tile
            folium.Rectangle([[ts, tw], [tn, te]], color="#00e5ff", weight=1.5, fill=False,
                             dash_array="6 4").add_to(rm)
        steger = res.get("steger")
        if steger is not None and len(steger):
            folium.GeoJson(steger, name="Steger lines (not used to split)", show=False,
                           style_function=lambda _: {"color": "#e6b800", "weight": 2.5}).add_to(rm)
        folium.Rectangle([[b[1], b[0]], [b[3], b[2]]], color="#00e5ff", weight=1, fill=False).add_to(rm)
        rm.fit_bounds([[b[1], b[0]], [b[3], b[2]]]); folium.LayerControl().add_to(rm)
        st_folium(rm, height=480, use_container_width=True,
                  key=f"result_map_{res['run_id']}_{int(straight)}_{tol_m}_{brightness:.1f}", returned_objects=[])
    with right:
        preview_cap = "Captured imagery (0.25 m) with detected field edges"
        if abs(brightness - 1.0) >= 1e-6:
            preview_cap += f". Screen brightness {brightness:.1f}; the PNG download is the original"
        st.image(D.view_rgb(res["preview"], brightness), caption=preview_cap, width="stretch")

    st.markdown("#### Downloads")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button("Fields (GeoJSON)", f.to_json(), "fields.geojson", "application/geo+json", width="stretch")
    d2.download_button("Boundaries (GeoJSON)", lines.to_json(), "field_boundaries.geojson",
                       "application/geo+json", width="stretch")
    steger = res.get("steger")
    if steger is not None and len(steger):
        d5.download_button("Steger lines (GeoJSON)", steger.to_json(), "steger_lines.geojson",
                           "application/geo+json", width="stretch")
    with tempfile.TemporaryDirectory() as tmp:
        gp = os.path.join(tmp, "field_edges.gpkg")
        if len(f):
            f.to_file(gp, layer="fields", driver="GPKG"); lines.to_file(gp, layer="boundaries", driver="GPKG")
            d3.download_button("GeoPackage (both layers)", open(gp, "rb").read(), "field_edges.gpkg",
                               "application/geopackage+sqlite3", width="stretch")
    ok, png = cv2.imencode(".png", cv2.cvtColor(res["preview"], cv2.COLOR_RGB2BGR))
    d4.download_button("Preview (PNG)", png.tobytes(), "field_edges_preview.png", "image/png", width="stretch")
    st.markdown("#### Field edges")
    c_st, c_tol = st.columns([1.3, 1])
    straight = c_st.checkbox(
        "Clean and straighten field edges",
        key="straighten_edges",
        help="Removes bends within the tolerance and replaces each side with one straight line. "
             "Shared corners of neighbouring fields snap together. The map, the boundary download, "
             "and the training shapefile use these edges.",
    )
    tol_m = c_tol.slider(
        "Tolerance (m)", min_value=1.0, max_value=12.0, value=3.0, step=0.5,
        key="straighten_tol_m", disabled=not straight,
        help="A bend, or a side shorter than this, is removed. A larger value leaves longer straight sides.",
    )
    if straight and len(res["fields"]):
        raw_n = _vertex_count(res["lines"])
        clean_n = _vertex_count(lines)
        st.caption(
            f"Sides are straight within {tol_m:.1f} m. Boundary vertices: {raw_n:,} → {clean_n:,}. "
            "The preview image remains the raster detection."
        )
    st.markdown("#### Training data (for retraining the models)")
    sample = res.get("tiles") or []
    if sample:
        st.caption(f"The rectangle was split into {len(sample)} sample tiles. Pick one to edit its GeoTIFF, "
                   "or download every tile in the training layout "
                   "(`Tiles/<name>_01.tif` and `Polyline/<name>_01.shp`).")
        tile_i = st.selectbox(
            "Sample tile", range(len(sample)),
            format_func=lambda i: f"Tile {i + 1} of {len(sample)} ({D.bbox_area_km2(*sample[i]['bbox']):.2f} km²)",
        )
        work_bbox = sample[tile_i]["bbox"]
        work_capture = (sample[tile_i]["rgb"], sample[tile_i]["transform"])
        tile_name_suffix = f"_{tile_i + 1:02d}"
    else:
        tile_i = 0
        work_bbox = res["bbox"]
        work_capture = res["capture"]
        tile_name_suffix = ""
        st.caption("The selected area as an RGB GeoTIFF at 0.25 m and the detected field edges as polylines, in the same "
                   "layout as `Labeled_Data/Data/<split>/<site>/{Tiles,Polyline}`. "
                   "Edit vertices on the GeoTIFF, then download the corrected lines for the training set.")
    crs_opts = {"WGS 84 (EPSG:4326)": "EPSG:4326",
                f"Local UTM ({D.utm_crs(res['bbox'])})": D.utm_crs(res["bbox"])}
    t0_, t1_ = st.columns([2, 3])
    crs = crs_opts[t0_.selectbox("Coordinate system", list(crs_opts))]
    name = t1_.text_input("Tile name", f"fe_{res['run_id']}").strip() or f"fe_{res['run_id']}"
    piece_name = f"{name}{tile_name_suffix}"
    tif_key = (res["run_id"], crs, tile_i)
    if st.session_state.get("tif_key") != tif_key:
        with st.spinner("Preparing GeoTIFF…"):
            st.session_state.tif_bytes = D.export_geotiff(*work_capture, work_bbox, crs)
            st.session_state.tif_key = tif_key
            st.session_state.tif_preview_key = None
    tif = st.session_state.tif_bytes
    edit_key = (res["run_id"], tile_i)
    geom_sig = (res["run_id"], bool(straight), round(float(tol_m), 2) if straight else None, tile_i)
    prev_sig = st.session_state.get("edge_geom_sig")
    if prev_sig not in (None, geom_sig) and prev_sig[0] == res["run_id"]:
        st.session_state.edge_edit_features = None
        st.session_state.edge_edit_run = None
    st.session_state.edge_geom_sig = geom_sig
    edited = st.session_state.get("edge_edit_features") if st.session_state.get("edge_edit_run") == edit_key else None
    predicted = D.features_from_lines(D.training_lines(lines, work_bbox, "EPSG:4326"))
    shown = edited if edited is not None else predicted
    tlines = D.lines_from_drawings(shown, crs)
    pack_key = (res["run_id"], crs, piece_name, _edit_token(edited), geom_sig[1], geom_sig[2], tile_i)
    if st.session_state.get("export_key") != pack_key:
        with st.spinner("Preparing polylines…"):
            shp = D.shapefile_bytes(tlines, piece_name)
        st.session_state.export = dict(tif=tif, shp=shp, n=len(tlines),
                                       pair=D.training_pair_zip(tif, shp, piece_name),
                                       edited=edited is not None)
        st.session_state.export_key = pack_key
    ex = st.session_state.export
    edit_on = st.checkbox(
        "Edit vertices on the GeoTIFF",
        key="edit_vertices",
        help="ArcGIS-style tools on the GeoTIFF: vertices, add, delete, move, continue, "
             "split, merge, reshape, new line, and delete line. Snapping, undo, and redo "
             "are on the same bar. Finish writes the lines into the shapefile.",
    )
    if ex.get("edited"):
        c_note, c_reset = st.columns([3, 1])
        c_note.caption(f"Shapefile uses your edited lines ({ex['n']}).")
        if c_reset.button("Reset to predicted lines", width="stretch"):
            st.session_state.edge_edit_features = None
            st.session_state.edge_edit_run = None
            st.rerun()
    if edit_on:
        if st.session_state.get("tif_preview_key") != tif_key:
            st.session_state.tif_preview = D.geotiff_display(tif)
            st.session_state.tif_preview_key = tif_key
        rgb, bounds = st.session_state.tif_preview
        rgb = D.view_rgb(rgb, brightness)
        st.caption(
            "The Edit bar works like an ArcGIS edit session. Vertices: drag a white handle, click a red "
            "midpoint to insert, right-click a vertex to delete. Also: Add vertex, Delete vertex, Move, "
            "Continue, Split, Merge, Reshape, New line, and Delete line. Snapping is on by default. "
            "Undo and Redo step through the sketch (Ctrl+Z / Ctrl+Y). Finish writes the lines into the shapefile."
        )
        drawn = st_folium(
            _edge_editor_map(rgb, bounds, shown),
            height=560, use_container_width=True,
            key=f"edge_edit_{res['run_id']}_{tile_i}_{int(straight)}_{tol_m}_{brightness:.1f}",
            returned_objects=["all_drawings"],
        )
        reported = D.normalize_drawings((drawn or {}).get("all_drawings"))
        if reported is not None and reported != shown:
            st.session_state.edge_edit_features = reported
            st.session_state.edge_edit_run = edit_key
            st.rerun()
    e1, e2, e3 = st.columns(3)
    e1.download_button(f"Image (GeoTIFF, {len(ex['tif']) / 1e6:.1f} MB)", ex["tif"], f"{piece_name}.tif", "image/tiff",
                       width="stretch")
    e2.download_button(f"Field edges (Shapefile, {ex['n']} lines)", ex["shp"], f"{piece_name}_polyline.zip",
                       "application/zip", width="stretch", disabled=ex["n"] == 0)
    if sample:
        zip_key = (res["run_id"], crs, name, bool(straight), round(float(tol_m), 2) if straight else None)
        if st.session_state.get("tiles_zip_key") != zip_key:
            with st.spinner(f"Preparing {len(sample)} sample tiles…"):
                st.session_state.tiles_zip = D.sample_tiles_zip(sample, lines, crs, name)
                st.session_state.tiles_zip_key = zip_key
        e3.download_button(f"All {len(sample)} sample tiles (ZIP)", st.session_state.tiles_zip,
                           f"{name}_sample_tiles.zip", "application/zip", type="primary", width="stretch")
    else:
        e3.download_button("Both, training layout (ZIP)", ex["pair"], f"{piece_name}_training.zip", "application/zip",
                           type="primary", width="stretch")

    with st.expander("Field table"):
        st.dataframe(pd.DataFrame(f.drop(columns="geometry")), width="stretch", hide_index=True)

st.markdown("---")
st.caption("Models trained and validated in `Labeled_Data/field_boundary_training.ipynb` on labelled "
           "Google-imagery screenshots from 10 sites in Bangladesh. Accuracy elsewhere may be lower.")
