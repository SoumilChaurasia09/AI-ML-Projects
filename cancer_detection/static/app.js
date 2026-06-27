// OncoVision Application Script

document.addEventListener("DOMContentLoaded", () => {
    // State management
    const activeDataset = "dermamnist";
    
    // UI Elements
    const noModelAlert = document.getElementById("no-model-alert");
    const evaluationContainer = document.getElementById("evaluation-container");
    const evalAccuracy = document.getElementById("eval-accuracy");
    const evalF1 = document.getElementById("eval-f1");
    const metricsTableBody = document.getElementById("metrics-table-body");
    const rocPlotImg = document.getElementById("roc-plot-img");
    const cmPlotImg = document.getElementById("cm-plot-img");

    const uploadZone = document.getElementById("upload-zone");
    const fileInput = document.getElementById("file-input");
    const inferenceResults = document.getElementById("inference-results");
    const previewContainer = document.getElementById("preview-container");
    const selectedImagePreview = document.getElementById("selected-image-preview");
    const btnCheckPrediction = document.getElementById("btn-check-prediction");
    const predictedClassName = document.getElementById("predicted-class-name");
    const probabilityBars = document.getElementById("probability-bars");
    
    let selectedFile = null;

    // Check if model results exist on disk (for persistence across restarts)
    const checkModelResults = async () => {
        try {
            const res = await fetch(`/api/model/results?dataset_name=${activeDataset}`);
            if (!res.ok) throw new Error("Failed to load model results.");
            const data = await res.json();
            
            if (data.has_model) {
                if (data.results) {
                    renderEvaluation(data.results);
                } else {
                    noModelAlert.classList.add("hidden");
                    evaluationContainer.classList.remove("hidden");
                }
            } else {
                hideEvaluation();
            }
        } catch (err) {
            console.error("Error checking model results:", err);
        }
    };

    const hideEvaluation = () => {
        noModelAlert.classList.remove("hidden");
        evaluationContainer.classList.add("hidden");
    };
    
    const renderEvaluation = (results) => {
        noModelAlert.classList.add("hidden");
        evaluationContainer.classList.remove("hidden");
        
        // Highlights
        evalAccuracy.textContent = `${(results.test_acc * 100).toFixed(1)}%`;
        const macroReport = results.classification_report["macro avg"] || results.classification_report["macro_avg"] || { "f1-score": 0 };
        evalF1.textContent = macroReport["f1-score"].toFixed(2);
        
        // Plots
        cmPlotImg.src = results.confusion_matrix_img;
        rocPlotImg.src = results.roc_curve_img;
        
        // Table body
        metricsTableBody.innerHTML = "";
        Object.entries(results.classification_report).forEach(([key, val]) => {
            // Skip avg summaries in main rows (we list them at bottom)
            if (key === "accuracy" || key === "macro avg" || key === "weighted avg") return;
            
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${key}</strong></td>
                <td>${(val.precision * 100).toFixed(1)}%</td>
                <td>${(val.recall * 100).toFixed(1)}%</td>
                <td>${val["f1-score"].toFixed(2)}</td>
                <td>${val.support}</td>
            `;
            metricsTableBody.appendChild(tr);
        });
        
        // Add averages as highlighted rows
        const macro = results.classification_report["macro_avg"] || results.classification_report["macro avg"];
        const weighted = results.classification_report["weighted_avg"] || results.classification_report["weighted avg"];
        
        if (macro && weighted) {
            const trMacro = document.createElement("tr");
            trMacro.style.borderTop = "2px solid rgba(255,255,255,0.1)";
            trMacro.innerHTML = `
                <td><em>Macro Avg</em></td>
                <td>${(macro.precision * 100).toFixed(1)}%</td>
                <td>${(macro.recall * 100).toFixed(1)}%</td>
                <td>${macro["f1-score"].toFixed(2)}</td>
                <td>${macro.support}</td>
            `;
            metricsTableBody.appendChild(trMacro);
            
            const trWeighted = document.createElement("tr");
            trWeighted.innerHTML = `
                <td><em>Weighted Avg</em></td>
                <td>${(weighted.precision * 100).toFixed(1)}%</td>
                <td>${(weighted.recall * 100).toFixed(1)}%</td>
                <td>${weighted["f1-score"].toFixed(2)}</td>
                <td>${weighted.support}</td>
            `;
            metricsTableBody.appendChild(trWeighted);
        }
    };

    // --- Diagnostic Inference (In-browser prediction) ---
    
    // Drag & Drop event bindings
    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.add("dragover");
        }, false);
    });
    
    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            uploadZone.classList.remove("dragover");
        }, false);
    });
    
    uploadZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processUpload(files[0]);
        }
    });
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            processUpload(e.target.files[0]);
        }
    });
    
    const processUpload = (file) => {
        selectedFile = file;
        
        // Hide previous classification outcome if visible
        if (inferenceResults) {
            inferenceResults.classList.add("hidden");
        }
        
        // Show Image Preview Container & Enable Check Button
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            if (selectedImagePreview) {
                selectedImagePreview.src = reader.result;
            }
            if (previewContainer) {
                previewContainer.classList.remove("hidden");
            }
        };
        
        if (btnCheckPrediction) {
            btnCheckPrediction.disabled = false;
            btnCheckPrediction.textContent = "🔍 Check Diagnostic Prediction";
        }
    };
    
    // Check prediction button listener
    if (btnCheckPrediction) {
        btnCheckPrediction.addEventListener("click", () => {
            if (selectedFile) {
                runInference(selectedFile);
            }
        });
    }
    
    const runInference = async (file) => {
        // Disable check button during analysis
        if (btnCheckPrediction) {
            btnCheckPrediction.disabled = true;
            btnCheckPrediction.textContent = "⚙️ Analyzing Scan...";
        }
        
        if (probabilityBars) {
            probabilityBars.innerHTML = `<p style="font-size:0.85rem;color:var(--text-secondary);">Calculating predictions...</p>`;
        }
        if (predictedClassName) {
            predictedClassName.textContent = "Analyzing...";
            predictedClassName.style.color = "var(--text-secondary)";
        }
        
        const formData = new FormData();
        formData.append("dataset_name", activeDataset);
        formData.append("file", file);
        
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });
            
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Failed to classify image.");
            }
            
            const result = await res.json();
            
            // Show result card
            if (inferenceResults) {
                inferenceResults.classList.remove("hidden");
            }
            
            // Display outputs
            if (predictedClassName) {
                predictedClassName.textContent = result.predicted_class;
                predictedClassName.style.color = "var(--success)";
            }
            
            if (probabilityBars) {
                probabilityBars.innerHTML = "";
                result.predictions.forEach(pred => {
                    const probPercent = (pred.probability * 100).toFixed(1);
                    
                    const item = document.createElement("div");
                    item.className = "prob-item";
                    item.innerHTML = `
                        <div class="prob-lbl">
                            <span>${pred.class_name}</span>
                            <span>${probPercent}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width: ${probPercent}%"></div>
                        </div>
                    `;
                    probabilityBars.appendChild(item);
                });
            }
            
        } catch (err) {
            console.error(err);
            if (predictedClassName) {
                predictedClassName.textContent = "Error";
                predictedClassName.style.color = "var(--danger)";
            }
            if (probabilityBars) {
                probabilityBars.innerHTML = `<p style="font-size:0.85rem;color:var(--danger);">Inference failed: ${err.message}</p>`;
            }
        } finally {
            // Restore check button state
            if (btnCheckPrediction) {
                btnCheckPrediction.disabled = false;
                btnCheckPrediction.textContent = "🔍 Check Diagnostic Prediction";
            }
        }
    };

    // --- Initial Load Setup ---
    checkModelResults();
});

