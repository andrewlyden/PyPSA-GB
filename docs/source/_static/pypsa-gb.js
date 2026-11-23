// Draw each Mermaid diagram at its natural size (its own font size), shrinking it only when it is
// wider than the page. sphinxcontrib-mermaid stretches every diagram to the full width at a fixed
// height, which made wide diagrams tiny; an inline !important max-width overrides that.
(function () {
  function fit() {
    document.querySelectorAll("pre.mermaid > svg").forEach(function (svg) {
      var box = svg.viewBox && svg.viewBox.baseVal;
      if (!box || !box.width) return;
      svg.style.setProperty("max-width", Math.ceil(box.width) + "px", "important");
      svg.style.setProperty("height", "auto", "important");
      svg.style.setProperty("display", "block");
      svg.style.setProperty("margin", "0 auto");
    });
  }
  new MutationObserver(fit).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", fit);
})();
