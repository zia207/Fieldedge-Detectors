"""Field Edge Detector — draw a box on Google / Esri satellite imagery and detect agricultural field
boundaries with models trained on labelled Google-imagery screenshots from Bangladesh.

Run:  cd FieldEdge_App && ../.venv/bin/streamlit run app.py
"""
import base64, hashlib, io, json, os, tempfile, time, zipfile

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
if CLOUD:
    MAX_KM2 = {"Delineate Anything V2 (fine-tuned) — fast": 0.05}
    MODEL_INFO = {
        "Delineate Anything V2 (fine-tuned) — fast":
            "Fine-tuned YOLO11x-seg. Held-out test: field F1 0.64, boundary F1 0.65 at 1 m. "
            "This public app runs on CPU, so keep the box under 0.05 km². The first visit downloads the weights.",
    }
else:
    MAX_KM2 = {
        "Delineate Anything V2 (fine-tuned) — fast": 2.0,
        "SAM 2.1 (pre-trained, calibrated) — most accurate": 0.25,
        "FC-DenseNet — boundary segmentation": 1.0,
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
TILE_ATTR = {"Google Satellite": "Imagery © Google", "Esri World Imagery": "Imagery © Esri, Maxar, Earthstar Geographics"}


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


def _edit_token(features):
    if features is None:
        return "predicted"
    blob = json.dumps(features, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


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
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fe-pygis"></div>', unsafe_allow_html=True)
    st.image(PYGIS_LOGO, width="stretch")
    st.title("🌾 Field Edge Detector")
    source = st.selectbox("Imagery", list(D.SOURCES), index=1 if CLOUD else 0,
                          help="Google is the imagery the models were trained on. Esri World Imagery is licence-friendlier.")
    scale = st.radio("Map scale / resolution", list(D.SCALES), index=0)
    zoom = D.SCALES[scale]
    if len(MODEL_INFO) == 1:
        model_name = next(iter(MODEL_INFO))
        st.markdown(f"**Model:** {model_name}")
    else:
        model_name = st.radio("Model", list(MODEL_INFO), index=0)
    st.caption(MODEL_INFO[model_name])
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
        auto_fallback = st.checkbox("Fall back to zoom 18 if zoom 19 has no imagery", value=True)
    st.markdown("---")
    where = "downloaded on first run" if CLOUD else "`Labeled_Data/`"
    st.caption(f"Compute device: **{D.DEVICE.upper()}** · weights {where}")
    if source == "Google Satellite":
        st.caption("⚠️ Google's Terms of Service restrict bulk download and reuse of imagery. "
                   "For research use only; use Esri World Imagery for shared or operational work.")
    st.markdown(
        f'''
        <div class="fe-brand-footer">
          <p>Zia Ahmed, PhD</p>
          <img src="data:image/png;base64,{_upatta_b64}" alt="Upatta Data Analytics">
        </div>
        ''',
        unsafe_allow_html=True,
    )

# ── map + drawing ────────────────────────────────────────────────────────
st.markdown("### 1 · Draw a rectangle around the fields")
st.caption("Use the ▭ tool (left side of the map). Pan or zoom first, or jump to coordinates below. "
           f"Maximum area for the selected model: **{MAX_KM2[model_name]} km²**.")

c1, c2, c3 = st.columns([1, 1, 1])
if "center" not in st.session_state:
    st.session_state.center = [22.315167, 90.154757]          # Bhola, Bangladesh
lat_in = c1.number_input("Latitude", value=float(st.session_state.center[0]), format="%.6f")
lon_in = c2.number_input("Longitude", value=float(st.session_state.center[1]), format="%.6f")
if c3.button("Go to coordinates", use_container_width=True):
    st.session_state.center = [lat_in, lon_in]

tiles = D.SOURCES[source].replace("{x}", "{x}").replace("{y}", "{y}").replace("{z}", "{z}")
m = folium.Map(location=st.session_state.center, zoom_start=17, max_zoom=20, tiles=None, control_scale=True)
folium.TileLayer(tiles=tiles, attr=TILE_ATTR[source], name=source, max_zoom=20, max_native_zoom=19).add_to(m)
Draw(draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False,
                   "circlemarker": False, "rectangle": {"shapeOptions": {"color": "#ffd400", "weight": 2}}},
     edit_options={"edit": False}).add_to(m)
Fullscreen().add_to(m)
out = st_folium(m, height=520, use_container_width=True, key=f"draw_{source}",
                returned_objects=["last_active_drawing", "all_drawings"])

bbox = None
drawn = (out or {}).get("last_active_drawing") or ((out or {}).get("all_drawings") or [None])[-1]
if drawn and drawn.get("geometry"):
    west, south, east, north = shape(drawn["geometry"]).bounds
    bbox = (west, south, east, north)

# ── run ──────────────────────────────────────────────────────────────────
st.markdown("### 2 · Detect field edges")
if bbox is None:
    st.info("Draw a rectangle on the map to enable detection.")
else:
    area = D.bbox_area_km2(*bbox)
    ok = area <= MAX_KM2[model_name]
    a1, a2, a3 = st.columns(3)
    a1.metric("Box area", f"{area:.3f} km²")
    a2.metric("Box size", f"{(bbox[2]-bbox[0])*111320*np.cos(np.radians((bbox[1]+bbox[3])/2)):.0f} × {(bbox[3]-bbox[1])*110570:.0f} m")
    a3.metric("Zoom", f"{zoom}")
    if not ok:
        st.error(f"The box is larger than {MAX_KM2[model_name]} km² for this model. Draw a smaller box "
                 + ("or switch to Delineate Anything." if "SAM" in model_name else "."))
    if st.button("🔍 Detect field edges", type="primary", disabled=not ok, use_container_width=True):
        prog = st.progress(0.0, text="Downloading imagery…"); t0 = time.time()
        rgb, tr, missing = D.capture(*bbox, zoom, source)
        used_zoom, note = zoom, ""
        if D.placeholder_fraction(rgb) > 0.5 and zoom == 19 and auto_fallback:
            rgb, tr, missing = D.capture(*bbox, 18, source)
            used_zoom, note = 18, f"{source} has no zoom-19 imagery here; zoom 18 was used instead."
        if D.placeholder_fraction(rgb) > 0.5:
            st.error(f"{source} has no imagery for this box at zoom {used_zoom}. Try the other source.")
            st.stop()
        img, tr_img, ground = D.to_training_resolution(rgb, tr, (bbox[1] + bbox[3]) / 2)
        prog.progress(0.1, text="Running model…")
        cb = lambda f, msg: prog.progress(0.1 + 0.75 * f, text=msg)
        if model_name.startswith("Delineate"):
            try:
                polys, scores = D.detect_delineate_anything(get_da(), img, patch=D.DA_SETTINGS["patch"], conf=da_conf, progress=cb)
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
            polys, scores = D.detect_sam2(get_sam2(), img, progress=cb, pred_iou_thresh=sam_iou)
        labels = D.polygons_to_labels(polys, img.shape[:2])
        n_snap, steger = 0, None
        if refine:
            refined, n_snap, curves = D.refine_outlines(
                img, labels, progress=lambda f, msg: prog.progress(0.86 + 0.10 * f, text=msg))
            steger = D.curves_to_gdf(curves, tr_img)
            labels = refined
        prog.progress(0.97, text="Building polygons…")
        fields, lines = D.to_vectors(polys, scores, tr_img, img.shape[:2], box(*bbox), labels=labels)
        preview = D.outline_preview(img, labels)
        st.session_state.result = dict(fields=fields, lines=lines, steger=steger, snapped=n_snap,
                                       refined=refine, preview=preview, image=img, capture=(rgb, tr), bbox=bbox, source=source,
                                       zoom=used_zoom, ground=ground, model=model_name, note=note, missing=missing,
                                       seconds=time.time() - t0, run_id=time.strftime("%Y%m%dT%H%M%S"))
        prog.progress(1.0, text="Done")

# ── results ──────────────────────────────────────────────────────────────
res = st.session_state.get("result")
if res:
    st.markdown("### 3 · Results")
    if res["note"]:
        st.warning(res["note"])
    if res["missing"]:
        st.warning(f"{res['missing']} imagery tiles could not be downloaded (shown black).")
    f = res["fields"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fields detected", f"{len(f)}")
    m2.metric("Median field size", f"{f.area_m2.median():.0f} m²" if len(f) else "–")
    m3.metric("Outlines snapped", f"{res.get('snapped', 0)}" if res.get("refined") else "off")
    m4.metric("Run time", f"{res['seconds']:.1f} s")
    scale_name = "20 m" if res["zoom"] == 19 else "50 m" if res["zoom"] == 18 else f"zoom {res['zoom']}"
    st.caption(
        f"Model: {res['model']}. {res['source']} at {scale_name} scale "
        f"({res['ground']:.2f} m/px), resampled to 0.25 m/px. "
        + ("Outlines were moved at most 0.75 m onto a Steger line centre. Interior lines were not used to split fields."
           if res.get("refined") else "Outlines are the model masks, without the snake.")
    )

    left, right = st.columns([3, 2])
    with left:
        b = res["bbox"]
        rm = folium.Map(location=[(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], tiles=None, max_zoom=20, control_scale=True)
        folium.TileLayer(tiles=D.SOURCES[res["source"]], attr=TILE_ATTR[res["source"]], name=res["source"],
                         max_zoom=20, max_native_zoom=19).add_to(rm)
        if len(f):
            folium.GeoJson(f, name="Fields", style_function=lambda _: {"color": "#ff2d2d", "weight": 1.5,
                                                                      "fillColor": "#ffd400", "fillOpacity": 0.12},
                           tooltip=folium.GeoJsonTooltip(["field_id", "area_m2", "score"],
                                                         aliases=["Field", "Area (m²)", "Score"])).add_to(rm)
        steger = res.get("steger")
        if steger is not None and len(steger):
            folium.GeoJson(steger, name="Steger lines (not used to split)", show=False,
                           style_function=lambda _: {"color": "#e6b800", "weight": 2.5}).add_to(rm)
        folium.Rectangle([[b[1], b[0]], [b[3], b[2]]], color="#00e5ff", weight=1, fill=False).add_to(rm)
        rm.fit_bounds([[b[1], b[0]], [b[3], b[2]]]); folium.LayerControl().add_to(rm)
        st_folium(rm, height=480, use_container_width=True, key="result_map", returned_objects=[])
    with right:
        st.image(res["preview"], caption="Captured imagery (0.25 m) with detected field edges", use_container_width=True)

    st.markdown("#### Downloads")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.download_button("Fields (GeoJSON)", f.to_json(), "fields.geojson", "application/geo+json", use_container_width=True)
    d2.download_button("Boundaries (GeoJSON)", res["lines"].to_json(), "field_boundaries.geojson",
                       "application/geo+json", use_container_width=True)
    steger = res.get("steger")
    if steger is not None and len(steger):
        d5.download_button("Steger lines (GeoJSON)", steger.to_json(), "steger_lines.geojson",
                           "application/geo+json", use_container_width=True)
    with tempfile.TemporaryDirectory() as tmp:
        gp = os.path.join(tmp, "field_edges.gpkg")
        if len(f):
            f.to_file(gp, layer="fields", driver="GPKG"); res["lines"].to_file(gp, layer="boundaries", driver="GPKG")
            d3.download_button("GeoPackage (both layers)", open(gp, "rb").read(), "field_edges.gpkg",
                               "application/geopackage+sqlite3", use_container_width=True)
    ok, png = cv2.imencode(".png", cv2.cvtColor(res["preview"], cv2.COLOR_RGB2BGR))
    d4.download_button("Preview (PNG)", png.tobytes(), "field_edges_preview.png", "image/png", use_container_width=True)
    st.markdown("#### Training data (for retraining the models)")
    st.caption("The selected area as an RGB GeoTIFF at 0.25 m and the detected field edges as polylines, in the same "
               "layout as `Labeled_Data/Data/<split>/<site>/{Tiles,Polyline}`. "
               "Edit vertices on the GeoTIFF, then download the corrected lines for the training set.")
    crs_opts = {"WGS 84 (EPSG:4326)": "EPSG:4326",
                f"Local UTM ({D.utm_crs(res['bbox'])})": D.utm_crs(res["bbox"])}
    t0_, t1_ = st.columns([2, 3])
    crs = crs_opts[t0_.selectbox("Coordinate system", list(crs_opts))]
    name = t1_.text_input("Tile name", f"fe_{res['run_id']}").strip() or f"fe_{res['run_id']}"
    tif_key = (res["run_id"], crs)
    if st.session_state.get("tif_key") != tif_key:
        with st.spinner("Preparing GeoTIFF…"):
            st.session_state.tif_bytes = D.export_geotiff(*res["capture"], res["bbox"], crs)
            st.session_state.tif_key = tif_key
            st.session_state.tif_preview_key = None
    tif = st.session_state.tif_bytes
    edited = st.session_state.get("edge_edit_features") if st.session_state.get("edge_edit_run") == res["run_id"] else None
    predicted = D.features_from_lines(D.training_lines(res["lines"], res["bbox"], "EPSG:4326"))
    shown = edited if edited is not None else predicted
    tlines = D.lines_from_drawings(shown, crs)
    pack_key = (res["run_id"], crs, name, _edit_token(edited))
    if st.session_state.get("export_key") != pack_key:
        with st.spinner("Preparing polylines…"):
            shp = D.shapefile_bytes(tlines, name)
        st.session_state.export = dict(tif=tif, shp=shp, n=len(tlines), pair=D.training_pair_zip(tif, shp, name),
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
        if c_reset.button("Reset to predicted lines", use_container_width=True):
            st.session_state.edge_edit_features = None
            st.session_state.edge_edit_run = None
            st.rerun()
    if edit_on:
        if st.session_state.get("tif_preview_key") != tif_key:
            st.session_state.tif_preview = D.geotiff_display(tif)
            st.session_state.tif_preview_key = tif_key
        rgb, bounds = st.session_state.tif_preview
        st.caption(
            "The Edit bar works like an ArcGIS edit session. Vertices: drag a white handle, click a red "
            "midpoint to insert, right-click a vertex to delete. Also: Add vertex, Delete vertex, Move, "
            "Continue, Split, Merge, Reshape, New line, and Delete line. Snapping is on by default. "
            "Undo and Redo step through the sketch (Ctrl+Z / Ctrl+Y). Finish writes the lines into the shapefile."
        )
        drawn = st_folium(
            _edge_editor_map(rgb, bounds, shown),
            height=560, use_container_width=True, key=f"edge_edit_{res['run_id']}",
            returned_objects=["all_drawings"],
        )
        reported = D.normalize_drawings((drawn or {}).get("all_drawings"))
        if reported is not None and reported != shown:
            st.session_state.edge_edit_features = reported
            st.session_state.edge_edit_run = res["run_id"]
            st.rerun()
    e1, e2, e3 = st.columns(3)
    e1.download_button(f"Image (GeoTIFF, {len(ex['tif']) / 1e6:.1f} MB)", ex["tif"], f"{name}.tif", "image/tiff",
                       use_container_width=True)
    e2.download_button(f"Field edges (Shapefile, {ex['n']} lines)", ex["shp"], f"{name}_polyline.zip",
                       "application/zip", use_container_width=True, disabled=ex["n"] == 0)
    e3.download_button("Both, training layout (ZIP)", ex["pair"], f"{name}_training.zip", "application/zip",
                       type="primary", use_container_width=True)

    with st.expander("Field table"):
        st.dataframe(pd.DataFrame(f.drop(columns="geometry")), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Models trained and validated in `Labeled_Data/field_boundary_training.ipynb` on labelled "
           "Google-imagery screenshots from 10 sites in Bangladesh. Accuracy elsewhere may be lower.")
