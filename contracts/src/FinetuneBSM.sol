// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { EnumerableSet } from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";
import { Initializable } from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import { UUPSUpgradeable } from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

/// @title FinetuneBSM
/// @notice Blueprint Service Manager for fine-tuning services.
/// @dev Operators register with GPU capabilities and supported fine-tuning methods.
///      Services only accept tsUSD payment (ShieldedCredits wrapped token) for anonymous billing.
contract FinetuneBSM is Initializable, UUPSUpgradeable, BlueprintServiceManagerBase {
    using EnumerableSet for EnumerableSet.AddressSet;

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

    error InvalidPaymentAsset(address asset);
    error InsufficientGpuCapability(uint32 required, uint32 provided);
    error MethodNotSupported(string method);
    error ModelNotSupported(string model);
    error OperatorNotRegistered(address operator);
    error EpochsExceedMax(uint32 requested, uint32 max);

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    event OperatorRegistered(address indexed operator, uint32 gpuCount, uint32 totalVramMib);
    event ModelConfigured(string model, uint32 minGpuVramMib, uint32 maxEpochs, uint64 pricePerEpoch);
    event FinetuneJobSubmitted(uint64 indexed serviceId, uint64 indexed jobCallId, string baseModel, string method);
    event FinetuneResultSubmitted(uint64 indexed serviceId, uint64 indexed jobCallId, string jobId, string status);

    // ═══════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Model pricing and requirements per model size tier
    struct ModelConfig {
        uint32 minGpuVramMib;
        uint32 maxEpochs;
        uint64 pricePerEpoch; // in tsUSD base units
        bool enabled;
    }

    /// @notice GPU and method capabilities reported by operator at registration
    struct OperatorCapabilities {
        string[] supportedMethods;
        string[] supportedModels;
        uint32 gpuCount;
        uint32 totalVramMib;
        string endpoint;
        bool active;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice The only accepted payment token (tsUSD -- shielded pool wrapped token)
    address public tsUSD;

    /// @notice Minimum operator stake (in TNT)
    uint256 public constant MIN_OPERATOR_STAKE = 100 ether;

    /// @notice operator => capabilities
    mapping(address => OperatorCapabilities) public operatorCaps;

    /// @notice model name hash => ModelConfig
    mapping(bytes32 => ModelConfig) public modelConfigs;

    /// @notice Set of registered operators
    EnumerableSet.AddressSet private _operators;

    // ═══════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @notice Initialize the contract (called once via proxy)
    /// @param _tsUSD The wrapped stablecoin accepted for payment
    function initialize(address _tsUSD) external initializer {
        __UUPSUpgradeable_init();
        tsUSD = _tsUSD;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Configure a model's pricing and requirements
    function configureModel(
        string calldata model,
        uint32 minGpuVramMib,
        uint32 maxEpochs,
        uint64 pricePerEpoch
    ) external onlyBlueprintOwner {
        bytes32 key = keccak256(bytes(model));
        modelConfigs[key] = ModelConfig({
            minGpuVramMib: minGpuVramMib,
            maxEpochs: maxEpochs,
            pricePerEpoch: pricePerEpoch,
            enabled: true
        });

        emit ModelConfigured(model, minGpuVramMib, maxEpochs, pricePerEpoch);
    }

    /// @notice Disable a model
    function disableModel(string calldata model) external onlyBlueprintOwner {
        bytes32 key = keccak256(bytes(model));
        modelConfigs[key].enabled = false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // OPERATOR LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Operator registers with GPU capabilities and supported methods/models.
    /// @param registrationInputs abi.encode(string[] supportedMethods, string[] supportedModels, uint32 gpuCount, uint32 totalVramMib, string endpoint)
    function onRegister(address operator, bytes calldata registrationInputs) external payable override onlyFromTangle {
        (
            string[] memory supportedMethods,
            string[] memory supportedModels,
            uint32 gpuCount,
            uint32 totalVramMib,
            string memory endpoint
        ) = abi.decode(registrationInputs, (string[], string[], uint32, uint32, string));

        operatorCaps[operator] = OperatorCapabilities({
            supportedMethods: supportedMethods,
            supportedModels: supportedModels,
            gpuCount: gpuCount,
            totalVramMib: totalVramMib,
            endpoint: endpoint,
            active: true
        });

        _operators.add(operator);

        emit OperatorRegistered(operator, gpuCount, totalVramMib);
    }

    function onUnregister(address operator) external override onlyFromTangle {
        operatorCaps[operator].active = false;
        _operators.remove(operator);
    }

    function onUpdatePreferences(address operator, bytes calldata newPreferences) external payable override onlyFromTangle {
        string memory newEndpoint = abi.decode(newPreferences, (string));
        operatorCaps[operator].endpoint = newEndpoint;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SERVICE LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    function onRequest(
        uint64,
        address,
        address[] calldata,
        bytes calldata,
        uint64,
        address paymentAsset,
        uint256
    ) external payable override onlyFromTangle {
        if (paymentAsset != tsUSD && paymentAsset != address(0)) {
            revert InvalidPaymentAsset(paymentAsset);
        }
    }

    function onServiceInitialized(
        uint64,
        uint64,
        uint64 serviceId,
        address,
        address[] calldata,
        uint64
    ) external override onlyFromTangle {
        _permitAsset(serviceId, tsUSD);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MULTI-OPERATOR DISTRIBUTED TRAINING
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Per-job contribution tracking for multi-operator training
    struct OperatorContribution {
        uint64 gpuHoursContributed;
        uint64 stepsCompleted;
        uint32 joinedAtEpoch;
        uint32 leftAtEpoch;
        bool active;
    }

    /// @notice Distributed training job state
    struct TrainingJobState {
        uint32 currentEpoch;
        uint32 totalEpochs;
        uint32 operatorCount;
        bytes32 latestCheckpointHash;
        uint64 totalPayment;
        bool completed;
    }

    /// @notice jobId => operator => contribution
    mapping(uint64 => mapping(address => OperatorContribution)) public jobContributions;

    /// @notice jobId => training state
    mapping(uint64 => TrainingJobState) public trainingJobs;

    /// @notice jobId => operators list
    mapping(uint64 => address[]) public jobOperators;

    event OperatorJoinedTraining(uint64 indexed jobId, address indexed operator, uint32 epoch);
    event OperatorLeftTraining(uint64 indexed jobId, address indexed operator, uint32 epoch);
    event CheckpointSubmitted(uint64 indexed jobId, bytes32 checkpointHash, uint32 epoch);
    event PaymentDistributed(uint64 indexed jobId, address indexed operator, uint64 amount);

    function canJoin(uint64, address operator) external view override returns (bool) {
        return operatorCaps[operator].active;
    }

    /// @notice Operator joins an active distributed training job
    function joinTraining(uint64 jobId) external {
        require(operatorCaps[msg.sender].active, "not registered");
        require(!jobContributions[jobId][msg.sender].active, "already in job");

        jobContributions[jobId][msg.sender] = OperatorContribution({
            gpuHoursContributed: 0,
            stepsCompleted: 0,
            joinedAtEpoch: trainingJobs[jobId].currentEpoch,
            leftAtEpoch: 0,
            active: true
        });
        jobOperators[jobId].push(msg.sender);
        trainingJobs[jobId].operatorCount++;

        emit OperatorJoinedTraining(jobId, msg.sender, trainingJobs[jobId].currentEpoch);
    }

    /// @notice Operator gracefully leaves a training job
    function leaveTraining(uint64 jobId) external {
        require(jobContributions[jobId][msg.sender].active, "not in job");
        jobContributions[jobId][msg.sender].active = false;
        jobContributions[jobId][msg.sender].leftAtEpoch = trainingJobs[jobId].currentEpoch;
        trainingJobs[jobId].operatorCount--;

        emit OperatorLeftTraining(jobId, msg.sender, trainingJobs[jobId].currentEpoch);
    }

    /// @notice Submit a checkpoint hash (any active operator in the job)
    function submitCheckpoint(uint64 jobId, bytes32 checkpointHash, uint32 epoch) external {
        require(jobContributions[jobId][msg.sender].active, "not in job");
        trainingJobs[jobId].latestCheckpointHash = checkpointHash;
        trainingJobs[jobId].currentEpoch = epoch;

        emit CheckpointSubmitted(jobId, checkpointHash, epoch);
    }

    /// @notice Record GPU-hours contribution (called via heartbeat validation)
    function recordContribution(uint64 jobId, address operator, uint64 gpuHours, uint64 steps) external onlyFromTangle {
        OperatorContribution storage c = jobContributions[jobId][operator];
        require(c.active, "not in job");
        c.gpuHoursContributed += gpuHours;
        c.stepsCompleted += steps;
    }

    /// @notice Distribute payment proportionally by GPU-hours contributed
    function getPaymentSplit(uint64 jobId) external view returns (address[] memory operators, uint64[] memory amounts) {
        address[] memory ops = jobOperators[jobId];
        uint64 totalHours = 0;
        for (uint i = 0; i < ops.length; i++) {
            totalHours += jobContributions[jobId][ops[i]].gpuHoursContributed;
        }
        if (totalHours == 0) return (ops, new uint64[](ops.length));

        uint64[] memory splits = new uint64[](ops.length);
        uint64 totalPayment = trainingJobs[jobId].totalPayment;
        for (uint i = 0; i < ops.length; i++) {
            splits[i] = (totalPayment * jobContributions[jobId][ops[i]].gpuHoursContributed) / totalHours;
        }
        return (ops, splits);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // JOB LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Validate a fine-tuning job submission
    /// @dev inputs = abi.encode(string baseModel, string datasetUrl, string method, uint32 epochs, uint32 batchSize)
    function onJobCall(
        uint64 serviceId,
        uint8,
        uint64 jobCallId,
        bytes calldata inputs
    ) external payable override onlyFromTangle {
        (string memory baseModel,, string memory method, uint32 epochs,) =
            abi.decode(inputs, (string, string, string, uint32, uint32));

        // Validate model is configured
        bytes32 modelKey = keccak256(bytes(baseModel));
        ModelConfig storage mc = modelConfigs[modelKey];
        if (!mc.enabled) revert ModelNotSupported(baseModel);

        // Validate epochs
        if (epochs > mc.maxEpochs) revert EpochsExceedMax(epochs, mc.maxEpochs);

        emit FinetuneJobSubmitted(serviceId, jobCallId, baseModel, method);
    }

    /// @notice Validate a fine-tuning job result
    /// @dev outputs = abi.encode(string jobId, string status, string adapterUrl)
    function onJobResult(
        uint64 serviceId,
        uint8,
        uint64 jobCallId,
        address operator,
        bytes calldata,
        bytes calldata outputs
    ) external payable override onlyFromTangle {
        if (!operatorCaps[operator].active) revert OperatorNotRegistered(operator);

        (string memory jobId, string memory status,) = abi.decode(outputs, (string, string, string));

        emit FinetuneResultSubmitted(serviceId, jobCallId, jobId, status);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // QUERIES
    // ═══════════════════════════════════════════════════════════════════════

    function queryIsPaymentAssetAllowed(uint64 serviceId, address asset) external view override returns (bool) {
        if (asset == address(0)) return true;
        address[] memory permitted = _getPermittedAssets(serviceId);
        if (permitted.length == 0) return asset == tsUSD;
        for (uint256 i; i < permitted.length; ++i) {
            if (permitted[i] == asset) return true;
        }
        return false;
    }

    function getAggregationThreshold(uint64, uint8) external pure override returns (uint16, uint8) {
        return (0, 0);
    }

    function getMinOperatorStake() external pure override returns (bool, uint256) {
        return (false, MIN_OPERATOR_STAKE);
    }

    function getHeartbeatInterval(uint64) external pure override returns (bool, uint64) {
        return (false, 100);
    }

    function getExitConfig(uint64) external pure override returns (bool, uint64, uint64, bool) {
        // 1 hour min commitment, 1 hour exit queue, force exit allowed
        return (false, 3600, 3600, true);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Get all registered operators
    function getOperators() external view returns (address[] memory) {
        return _operators.values();
    }

    /// @notice Get operator count
    function getOperatorCount() external view returns (uint256) {
        return _operators.length();
    }

    /// @notice Get model config by name
    function getModelConfig(string calldata model) external view returns (ModelConfig memory) {
        return modelConfigs[keccak256(bytes(model))];
    }

    /// @notice Check if an operator is registered and active
    function isOperatorActive(address operator) external view returns (bool) {
        return operatorCaps[operator].active;
    }

    /// @notice Get operator pricing for a given operator address.
    function getOperatorPricing(address operator)
        external
        view
        returns (uint64 pricePerEpoch, string memory endpoint)
    {
        OperatorCapabilities storage caps = operatorCaps[operator];
        if (!caps.active) revert OperatorNotRegistered(operator);

        if (caps.supportedModels.length > 0) {
            bytes32 modelKey = keccak256(bytes(caps.supportedModels[0]));
            ModelConfig storage mc = modelConfigs[modelKey];
            return (mc.pricePerEpoch, caps.endpoint);
        }
        return (0, caps.endpoint);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // UPGRADES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Only the blueprint owner can authorize upgrades
    function _authorizeUpgrade(address) internal override onlyBlueprintOwner {}

    receive() external payable override {}
}
