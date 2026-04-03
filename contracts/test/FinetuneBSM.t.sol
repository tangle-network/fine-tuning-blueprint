// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Test } from "forge-std/Test.sol";
import { ERC1967Proxy } from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import { FinetuneBSM } from "../src/FinetuneBSM.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

contract FinetuneBSMTest is Test {
    FinetuneBSM public bsm;
    address public tsUSD;

    address public tangleCore = address(0xC0DE);
    address public owner = address(0xBEEF);
    address public operator1 = address(0x1111);
    address public operator2 = address(0x2222);
    address public user = address(0x3333);

    function setUp() public {
        tsUSD = address(0xAAAA);

        FinetuneBSM impl_ = new FinetuneBSM();
        bytes memory initData = abi.encodeCall(FinetuneBSM.initialize, (tsUSD));
        ERC1967Proxy proxy = new ERC1967Proxy(address(impl_), initData);
        bsm = FinetuneBSM(payable(address(proxy)));

        bsm.onBlueprintCreated(1, owner, tangleCore);

        vm.prank(owner);
        bsm.configureModel(
            "meta-llama/Llama-3.1-8B-Instruct",
            uint32(16_000), // min VRAM
            uint32(10),     // max epochs
            uint64(500_000) // price per epoch
        );
    }

    // --- Initialization ---

    function test_initialization() public view {
        assertEq(bsm.blueprintId(), 1);
        assertEq(bsm.blueprintOwner(), owner);
        assertEq(bsm.tangleCore(), tangleCore);
        assertEq(bsm.tsUSD(), tsUSD);
    }

    function test_cannotReinitialize() public {
        vm.expectRevert(BlueprintServiceManagerBase.AlreadyInitialized.selector);
        bsm.onBlueprintCreated(2, owner, tangleCore);
    }

    // --- Model Configuration ---

    function test_configureModel() public view {
        FinetuneBSM.ModelConfig memory mc = bsm.getModelConfig("meta-llama/Llama-3.1-8B-Instruct");
        assertEq(mc.minGpuVramMib, 16_000);
        assertEq(mc.maxEpochs, 10);
        assertEq(mc.pricePerEpoch, 500_000);
        assertTrue(mc.enabled);
    }

    function test_configureModel_onlyOwner() public {
        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, user, owner)
        );
        bsm.configureModel("test-model", 8000, 5, 100_000);
    }

    function test_disableModel() public {
        vm.prank(owner);
        bsm.disableModel("meta-llama/Llama-3.1-8B-Instruct");

        FinetuneBSM.ModelConfig memory mc = bsm.getModelConfig("meta-llama/Llama-3.1-8B-Instruct");
        assertFalse(mc.enabled);
    }

    // --- Operator Registration ---

    function test_registerOperator() public {
        _registerOperator(operator1);

        assertTrue(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 1);
    }

    function test_registerOperator_onlyTangle() public {
        string[] memory methods = new string[](1);
        methods[0] = "lora";
        string[] memory models = new string[](1);
        models[0] = "meta-llama/Llama-3.1-8B-Instruct";

        bytes memory regData = abi.encode(methods, models, uint32(2), uint32(48_000), "https://op1.example.com");

        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, user, tangleCore)
        );
        bsm.onRegister(operator1, regData);
    }

    // --- Unregistration ---

    function test_unregisterOperator() public {
        _registerOperator(operator1);

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertFalse(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 0);
    }

    // --- Pricing ---

    function test_getOperatorPricing() public {
        _registerOperator(operator1);

        (uint64 pricePerEpoch, string memory endpoint) = bsm.getOperatorPricing(operator1);
        assertEq(pricePerEpoch, 500_000);
        assertEq(keccak256(bytes(endpoint)), keccak256(bytes("https://op1.example.com")));
    }

    function test_getOperatorPricing_unregistered() public {
        vm.expectRevert(
            abi.encodeWithSelector(FinetuneBSM.OperatorNotRegistered.selector, operator1)
        );
        bsm.getOperatorPricing(operator1);
    }

    // --- Job Lifecycle ---

    function test_onJobCall() public {
        bytes memory inputs = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct",
            "https://example.com/dataset.jsonl",
            "lora",
            uint32(5),
            uint32(8)
        );

        vm.prank(tangleCore);
        bsm.onJobCall(1, 0, 1, inputs);
    }

    function test_onJobCall_unsupportedModel() public {
        bytes memory inputs = abi.encode(
            "nonexistent-model",
            "https://example.com/dataset.jsonl",
            "lora",
            uint32(5),
            uint32(8)
        );

        vm.prank(tangleCore);
        vm.expectRevert(
            abi.encodeWithSelector(FinetuneBSM.ModelNotSupported.selector, "nonexistent-model")
        );
        bsm.onJobCall(1, 0, 1, inputs);
    }

    function test_onJobCall_epochsExceedMax() public {
        bytes memory inputs = abi.encode(
            "meta-llama/Llama-3.1-8B-Instruct",
            "https://example.com/dataset.jsonl",
            "lora",
            uint32(20), // exceeds max of 10
            uint32(8)
        );

        vm.prank(tangleCore);
        vm.expectRevert(
            abi.encodeWithSelector(FinetuneBSM.EpochsExceedMax.selector, 20, 10)
        );
        bsm.onJobCall(1, 0, 1, inputs);
    }

    function test_onJobResult() public {
        _registerOperator(operator1);

        bytes memory inputs = "";
        bytes memory outputs = abi.encode("job-123", "completed", "https://example.com/adapter.safetensors");

        vm.prank(tangleCore);
        bsm.onJobResult(1, 0, 1, operator1, inputs, outputs);
    }

    function test_onJobResult_unregisteredOperator() public {
        bytes memory outputs = abi.encode("job-123", "completed", "https://example.com/adapter.safetensors");

        vm.prank(tangleCore);
        vm.expectRevert(
            abi.encodeWithSelector(FinetuneBSM.OperatorNotRegistered.selector, operator1)
        );
        bsm.onJobResult(1, 0, 1, operator1, "", outputs);
    }

    // --- Distributed Training ---

    function test_joinTraining() public {
        _registerOperator(operator1);

        vm.prank(operator1);
        bsm.joinTraining(1);

        // Verify contribution is recorded
        (uint64 gpuHours, uint64 steps, uint32 joinedAt, uint32 leftAt, bool active) =
            bsm.jobContributions(1, operator1);
        assertTrue(active);
        assertEq(gpuHours, 0);
        assertEq(steps, 0);
    }

    function test_leaveTraining() public {
        _registerOperator(operator1);

        vm.prank(operator1);
        bsm.joinTraining(1);

        vm.prank(operator1);
        bsm.leaveTraining(1);

        (,,,, bool active) = bsm.jobContributions(1, operator1);
        assertFalse(active);
    }

    function test_submitCheckpoint() public {
        _registerOperator(operator1);

        vm.prank(operator1);
        bsm.joinTraining(1);

        vm.prank(operator1);
        bsm.submitCheckpoint(1, keccak256("checkpoint-data"), 3);

        (uint32 currentEpoch,,,,,) = bsm.trainingJobs(1);
        assertEq(currentEpoch, 3);
    }

    // --- Configuration Queries ---

    function test_minOperatorStake() public view {
        (bool useDefault, uint256 minStake) = bsm.getMinOperatorStake();
        assertFalse(useDefault);
        assertEq(minStake, 100 ether);
    }

    function test_canJoin() public {
        _registerOperator(operator1);
        assertTrue(bsm.canJoin(1, operator1));
        assertFalse(bsm.canJoin(1, operator2));
    }

    // --- Helpers ---

    function _registerOperator(address op) internal {
        string[] memory methods = new string[](2);
        methods[0] = "lora";
        methods[1] = "qlora";
        string[] memory models = new string[](1);
        models[0] = "meta-llama/Llama-3.1-8B-Instruct";

        bytes memory regData = abi.encode(methods, models, uint32(2), uint32(48_000), "https://op1.example.com");
        vm.prank(tangleCore);
        bsm.onRegister(op, regData);
    }
}
