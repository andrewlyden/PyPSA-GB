document.addEventListener('DOMContentLoaded', function() {
    // Load the GuruBase widget
    const guruScript = document.createElement("script");
    guruScript.src = "https://widget.gurubase.io/widget.latest.min.js";
    guruScript.async = true;
    guruScript.id = "guru-widget-id";

    // Widget configuration
    guruScript.setAttribute("data-widget-id", "NCLcyH19q0SWqySmr4oLdrAjp-0S87TnlVeDAu7AxEk");
    guruScript.setAttribute("data-text", "Ask AI");
    guruScript.setAttribute("data-margins", '{"bottom": "1rem", "right": "1rem"}');
    guruScript.setAttribute("data-light-mode", "auto");
    guruScript.setAttribute("data-window-mode", "sidebar");
    guruScript.setAttribute("data-overlap-content", "true");
    guruScript.setAttribute("data-bg-color", "#86AE9C");
    guruScript.setAttribute("data-shimmer-color", "#86AE9C");

    // Append the script to the document
    document.body.appendChild(guruScript);
});