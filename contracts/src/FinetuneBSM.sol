// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @title FinetuneBSM — Blueprint Service Manager for model fine-tuning
/// @dev Operators run LoRA/QLoRA/full fine-tuning on GPU hardware.
/// Pricing: per-epoch per model size tier. Async job model.

import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

contract FinetuneBSM is BlueprintServiceManagerBase {
    struct OperatorCapabilities {
        string[] supportedMethods;  // ["lora", "qlora", "full"]
        string[] supportedModels;   // ["llama-3.1-8b", "mistral-7b", ...]
        uint32 gpuCount;
        uint32 totalVramMib;
        string gpuModel;
        string endpoint;
        bool active;
    }

    struct ModelTier {
        uint64 pricePerEpoch;       // in payment token base units
        uint32 minGpuVramMib;
        uint32 maxEpochs;
        bool enabled;
    }

    mapping(address => OperatorCapabilities) public operatorCaps;
    mapping(bytes32 => ModelTier) public modelTiers;
    address[] internal _operators;

    function configureModelTier(
        string calldata modelPattern,
        uint64 pricePerEpoch,
        uint32 minGpuVramMib,
        uint32 maxEpochs
    ) external onlyBlueprintOwner {
        modelTiers[keccak256(bytes(modelPattern))] = ModelTier(pricePerEpoch, minGpuVramMib, maxEpochs, true);
    }

    function onRegister(address operator, bytes calldata registrationInputs) external payable override onlyFromTangle {
        (
            string[] memory methods,
            string[] memory models,
            uint32 gpuCount,
            uint32 totalVramMib,
            string memory gpuModel,
            string memory endpoint
        ) = abi.decode(registrationInputs, (string[], string[], uint32, uint32, string, string));
        operatorCaps[operator] = OperatorCapabilities(methods, models, gpuCount, totalVramMib, gpuModel, endpoint, true);
        _operators.push(operator);
    }

    function calculateCost(string calldata modelTier, uint32 epochs) external view returns (uint64) {
        ModelTier memory tier = modelTiers[keccak256(bytes(modelTier))];
        require(tier.enabled, "Model tier not configured");
        require(epochs <= tier.maxEpochs, "Exceeds max epochs");
        return tier.pricePerEpoch * uint64(epochs);
    }

    function getOperators() external view returns (address[] memory) { return _operators; }
    function isOperatorActive(address op) external view returns (bool) { return operatorCaps[op].active; }
    function getOperatorPricing(address op) external view returns (string memory endpoint) {
        return operatorCaps[op].endpoint;
    }
    function onUnregister(address operator) external override onlyFromTangle { operatorCaps[operator].active = false; }
    function onUpdatePreferences(address operator, bytes calldata p) external payable override onlyFromTangle {
        operatorCaps[operator].endpoint = abi.decode(p, (string));
    }
}
