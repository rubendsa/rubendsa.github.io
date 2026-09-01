(function () {
  "use strict";

  var steppers = document.querySelectorAll("[data-ppo-stepper]");
  if (!steppers.length) {
    return;
  }

  var envOptions = [512, 1024, 2048, 4096, 8192];
  var horizonOptions = [8, 16, 24, 32, 48];
  var numberFormat = new Intl.NumberFormat("en-US");
  var stageNames = ["Observe", "Act + step", "Store", "Advantage", "Sample", "Optimize", "Repeat"];

  function each(items, callback) {
    Array.prototype.forEach.call(items, callback);
  }

  function setText(selector, value) {
    each(document.querySelectorAll(selector), function (node) {
      node.textContent = value;
    });
  }

  each(steppers, function (root) {
    var tabs = root.querySelectorAll("[data-ppo-step]");
    var panels = root.querySelectorAll("[data-ppo-panel]");
    var previous = root.querySelector("[data-ppo-prev]");
    var next = root.querySelector("[data-ppo-next]");
    var counter = root.querySelector("[data-ppo-counter]");
    var status = root.querySelector("[data-ppo-status]");
    var stageName = root.querySelector("[data-ppo-stage-name]");
    var envInput = root.querySelector("[data-ppo-env-input]");
    var horizonInput = root.querySelector("[data-ppo-horizon-input]");
    var envOutput = root.querySelector("[data-ppo-env-output]");
    var horizonOutput = root.querySelector("[data-ppo-horizon-output]");
    var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var currentStage = 0;

    function updateParameters() {
      var envCount = envOptions[Number(envInput.value)];
      var horizon = horizonOptions[Number(horizonInput.value)];
      var transitions = envCount * horizon;
      var miniBatch = transitions / 4;

      envOutput.textContent = numberFormat.format(envCount);
      horizonOutput.textContent = numberFormat.format(horizon);
      envInput.setAttribute("aria-valuetext", numberFormat.format(envCount) + " parallel environments");
      horizonInput.setAttribute("aria-valuetext", numberFormat.format(horizon) + " rollout steps");

      setText("[data-ppo-envs]", numberFormat.format(envCount));
      setText("[data-ppo-horizon]", numberFormat.format(horizon));
      setText("[data-ppo-transitions]", numberFormat.format(transitions));
      setText("[data-ppo-minibatch]", numberFormat.format(miniBatch));
    }

    function centerActiveTab() {
      var activeTab = tabs[currentStage];
      if (activeTab && activeTab.scrollIntoView) {
        activeTab.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest", inline: "center" });
      }
    }

    function revealActivePanel() {
      var activePanel = panels[currentStage];
      if (!activePanel) {
        return;
      }

      window.requestAnimationFrame(function () {
        activePanel.focus({ preventScroll: true });
        if (activePanel.scrollIntoView) {
          activePanel.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
        }
      });
    }

    function showStage(index, announce, revealPanel) {
      currentStage = Math.max(0, Math.min(index, panels.length - 1));

      each(tabs, function (tab, tabIndex) {
        var active = tabIndex === currentStage;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", active ? "true" : "false");
        tab.setAttribute("tabindex", active ? "0" : "-1");
      });

      each(panels, function (panel, panelIndex) {
        var active = panelIndex === currentStage;
        panel.classList.toggle("is-active", active);
        panel.hidden = !active;
      });

      counter.textContent = String(currentStage + 1).padStart(2, "0") + " / " + String(panels.length).padStart(2, "0");
      stageName.textContent = stageNames[currentStage];
      previous.disabled = currentStage === 0;

      if (currentStage === panels.length - 1) {
        next.textContent = "Start a fresh rollout ↻";
      } else {
        next.textContent = "Next: " + stageNames[currentStage + 1] + " →";
      }

      if (announce) {
        status.textContent = "Stage " + (currentStage + 1) + " of " + panels.length + ": " + stageNames[currentStage];
        centerActiveTab();
        if (revealPanel) {
          revealActivePanel();
        }
      }
    }

    each(tabs, function (tab, index) {
      tab.id = "ppo-stage-tab-" + index;
      panels[index].setAttribute("aria-labelledby", tab.id);

      tab.addEventListener("click", function () {
        showStage(index, true, false);
      });

      tab.addEventListener("keydown", function (event) {
        var destination = index;

        if (event.key === "ArrowRight") {
          destination = (index + 1) % tabs.length;
        } else if (event.key === "ArrowLeft") {
          destination = (index - 1 + tabs.length) % tabs.length;
        } else if (event.key === "Home") {
          destination = 0;
        } else if (event.key === "End") {
          destination = tabs.length - 1;
        } else {
          return;
        }

        event.preventDefault();
        showStage(destination, true, false);
        tabs[destination].focus();
      });
    });

    previous.addEventListener("click", function () {
      showStage(currentStage - 1, true, true);
    });

    next.addEventListener("click", function () {
      showStage(currentStage === panels.length - 1 ? 0 : currentStage + 1, true, true);
    });

    root.addEventListener("keydown", function (event) {
      var tagName = event.target.tagName;
      if (tagName === "INPUT" || tagName === "BUTTON") {
        return;
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();
        showStage(currentStage === panels.length - 1 ? 0 : currentStage + 1, true, true);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        showStage(currentStage === 0 ? panels.length - 1 : currentStage - 1, true, true);
      }
    });

    envInput.addEventListener("input", updateParameters);
    horizonInput.addEventListener("input", updateParameters);

    root.setAttribute("data-ppo-enhanced", "true");
    updateParameters();
    showStage(0, false, false);
  });
})();
