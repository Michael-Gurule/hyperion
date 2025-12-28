"""
Test detection and sensor fusion capabilities.
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.detection import ThreatDetector, SensorFusion, MultiTargetTracker


def test_sensor_fusion():
    """Test Kalman filter sensor fusion."""
    print("Testing sensor fusion...")

    # Initialize fusion filter
    fusion = SensorFusion(state_dim=4, process_noise=0.1, measurement_noise=1.0)

    # Simulate target moving in straight line
    true_position = np.array([1000.0, 1000.0])
    true_velocity = np.array([100.0, 50.0])

    dt = 0.1
    num_steps = 50

    estimated_positions = []
    true_positions = []

    for step in range(num_steps):
        # Update true position
        true_position += true_velocity * dt
        true_positions.append(true_position.copy())

        # Simulate noisy measurement
        measurement_noise = np.random.normal(0, 10.0, 2)
        measurement = true_position + measurement_noise

        # Update fusion filter
        fusion.predict(dt)
        fusion.update(measurement)

        # Get estimate
        estimated_pos = fusion.get_position()
        estimated_positions.append(estimated_pos.copy())

    # Calculate estimation error
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)

    errors = np.linalg.norm(true_positions - estimated_positions, axis=1)
    mean_error = np.mean(errors)

    print(f"✓ Sensor fusion test complete")
    print(f"  Mean position error: {mean_error:.2f} meters")
    print(f"  Final position error: {errors[-1]:.2f} meters")
    print(f"  Final velocity estimate: {fusion.get_velocity()}")
    print(f"  True velocity: {true_velocity}")

    return mean_error < 20.0  # Should be better than raw measurements


def test_multi_sensor_fusion():
    """Test fusion of RF and thermal measurements."""
    print("\nTesting multi-sensor fusion...")

    fusion = SensorFusion()

    # Simulate target moving
    true_position = np.array([5000.0, 3000.0])
    true_velocity = np.array([100.0, 50.0])

    dt = 0.1
    num_steps = 10

    errors = []

    for step in range(num_steps):
        # Update true position
        true_position += true_velocity * dt

        # RF sensor (less accurate)
        rf_noise = np.random.normal(0, 50.0, 2)
        rf_measurement = true_position + rf_noise
        rf_confidence = 0.6

        # Thermal sensor (more accurate)
        thermal_noise = np.random.normal(0, 20.0, 2)
        thermal_measurement = true_position + thermal_noise
        thermal_confidence = 0.9

        # Predict
        if step > 0:
            fusion.predict(dt)

        # Fuse measurements
        fused_pos, confidence = fusion.fuse_measurements(
            rf_measurement, thermal_measurement, rf_confidence, thermal_confidence
        )

        error = np.linalg.norm(fused_pos - true_position)
        errors.append(error)

    # Check final error (after filter has converged)
    final_error = errors[-1]
    mean_error = np.mean(errors[-5:])  # Average of last 5 steps

    print(f"✓ Multi-sensor fusion test complete")
    print(f"  True final position: {true_position}")
    print(f"  Fused final position: {fused_pos}")
    print(f"  Initial error: {errors[0]:.2f} meters")
    print(f"  Final error: {final_error:.2f} meters")
    print(f"  Mean error (last 5 steps): {mean_error:.2f} meters")
    print(f"  Overall confidence: {confidence:.2f}")

    # Pass if converged error is reasonable
    return mean_error < 100.0  # More lenient threshold after convergence


def test_threat_detector():
    """Test neural network threat detector."""
    print("\nTesting threat detector network...")

    # Create detector
    detector = ThreatDetector(
        rf_feature_dim=128,
        thermal_feature_dim=128,
        hidden_dim=256,
        num_threat_classes=3,
    )

    # Generate random features
    batch_size = 10
    rf_features = torch.randn(batch_size, 128)
    thermal_features = torch.randn(batch_size, 128)

    # Forward pass
    class_logits, confidence = detector(rf_features, thermal_features)

    print(f"✓ Threat detector test complete")
    print(f"  Input batch size: {batch_size}")
    print(f"  Output class logits shape: {class_logits.shape}")
    print(f"  Output confidence shape: {confidence.shape}")
    print(f"  Sample class prediction: {torch.argmax(class_logits[0]).item()}")
    print(f"  Sample confidence: {confidence[0].item():.3f}")

    # Test detection method
    rf_sample = np.random.randn(128)
    thermal_sample = np.random.randn(128)

    detected, threat_class, conf = detector.detect(
        rf_sample, thermal_sample, threshold=0.5
    )

    print(f"\n  Detection test:")
    print(f"    Detected: {detected}")
    print(f"    Threat class: {threat_class}")
    print(f"    Confidence: {conf:.3f}")

    return class_logits.shape == (batch_size, 3) and confidence.shape == (batch_size, 1)


def test_multi_target_tracker():
    """Test multi-target tracking."""
    print("\nTesting multi-target tracker...")

    tracker = MultiTargetTracker(
        max_tracks=5, association_threshold=500.0, track_confirmation_threshold=3
    )

    # Simulate two targets
    target1_pos = np.array([1000.0, 1000.0])
    target1_vel = np.array([100.0, 50.0])

    target2_pos = np.array([2000.0, 3000.0])
    target2_vel = np.array([80.0, -60.0])

    dt = 0.1
    num_steps = 20

    confirmed_tracks_history = []

    for step in range(num_steps):
        # Update true positions
        target1_pos += target1_vel * dt
        target2_pos += target2_vel * dt

        # Add measurement noise
        meas1 = target1_pos + np.random.normal(0, 10.0, 2)
        meas2 = target2_pos + np.random.normal(0, 10.0, 2)

        # Update tracker
        measurements = [meas1, meas2]
        confirmed_tracks = tracker.update(measurements, dt)

        confirmed_tracks_history.append(len(confirmed_tracks))

    final_tracks = tracker.get_tracks()

    print(f"✓ Multi-target tracker test complete")
    print(f"  Total tracks created: {tracker.next_track_id}")
    print(f"  Final confirmed tracks: {len(final_tracks)}")
    print(f"  Track confirmation progression: {confirmed_tracks_history}")

    if len(final_tracks) > 0:
        for i, track in enumerate(final_tracks):
            pos = track["filter"].get_position()
            vel = track["filter"].get_velocity()
            print(f"  Track {i}: pos={pos}, vel={vel}, hits={track['hits']}")

    return len(final_tracks) >= 1  # Should have at least one confirmed track


def test_detection_integration():
    """Test integrated detection pipeline."""
    print("\nTesting integrated detection pipeline...")

    # Create components
    detector = ThreatDetector()
    fusion = SensorFusion()
    tracker = MultiTargetTracker()

    # Simulate detection scenario
    num_steps = 30
    dt = 0.1

    true_target_pos = np.array([8000.0, 6000.0])
    true_target_vel = np.array([200.0, 100.0])

    detections = 0

    for step in range(num_steps):
        # Update true position
        true_target_pos += true_target_vel * dt

        # Simulate sensor readings
        rf_features = np.random.randn(128)
        thermal_features = np.random.randn(128)

        # Detect threat
        detected, threat_class, confidence = detector.detect(
            rf_features, thermal_features, threshold=0.3
        )

        if detected:
            detections += 1

            # Generate position measurements
            rf_pos = true_target_pos + np.random.normal(0, 50.0, 2)
            thermal_pos = true_target_pos + np.random.normal(0, 30.0, 2)

            # Fuse measurements
            fused_pos, fused_conf = fusion.fuse_measurements(
                rf_pos,
                thermal_pos,
                rf_confidence=confidence * 0.7,
                thermal_confidence=confidence * 0.9,
            )

            # Update tracker
            confirmed_tracks = tracker.update([fused_pos], dt)

    final_tracks = tracker.get_tracks()

    print(f"✓ Integration test complete")
    print(f"  Total steps: {num_steps}")
    print(f"  Detections: {detections}")
    print(f"  Confirmed tracks: {len(final_tracks)}")

    if len(final_tracks) > 0:
        track_pos = final_tracks[0]["filter"].get_position()
        track_vel = final_tracks[0]["filter"].get_velocity()
        pos_error = np.linalg.norm(track_pos - true_target_pos)
        vel_error = np.linalg.norm(track_vel - true_target_vel)

        print(f"  Final position error: {pos_error:.2f} meters")
        print(f"  Final velocity error: {vel_error:.2f} m/s")

    return detections > 0 and len(final_tracks) > 0


def run_all_tests():
    """Execute all detection tests."""
    print("=" * 60)
    print("HYPERION Detection Module Test Suite")
    print("=" * 60)

    results = []

    try:
        results.append(("Sensor Fusion", test_sensor_fusion()))
        results.append(("Multi-Sensor Fusion", test_multi_sensor_fusion()))
        results.append(("Threat Detector", test_threat_detector()))
        results.append(("Multi-Target Tracker", test_multi_target_tracker()))
        results.append(("Detection Integration", test_detection_integration()))

        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)

        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name}: {status}")

        all_passed = all(result[1] for result in results)

        if all_passed:
            print("\n" + "=" * 60)
            print("All detection tests passed!")
            print("=" * 60)
        else:
            print("\nSome tests failed. Please review.")

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
